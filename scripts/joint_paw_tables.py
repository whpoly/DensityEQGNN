"""Dataset-specific PAW tables for the joint NMC and EC experiment."""
import re
import zipfile
import hashlib
from pathlib import Path
import torch
import numpy as np
from scipy.interpolate import CubicSpline
from scripts.deepdft_paw import BLOCKS, WIDTH, ELEMENTS
from scripts.evaluate_vasp_sad_nmc import AtomicFormFactor, POTCAR_PREFIX, POTCAR_ELEMENTS, TABLE_SIZE, _floats_after


def potcar_specs(path=Path('potpaw54.zip'), elements=None):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for symbol,z in (ELEMENTS if elements is None else elements).items():
            name = f'potpaw54/potpaw_PBE/PAW_GGA_PBE/{symbol}/POTCAR'
            raw = archive.read(name)
            text = raw.decode()
            basis = []
            for l,n in re.findall(r'Non local Part\s*\n\s*(\d+)\s+(\d+)',text):
                basis.extend([int(l)]*int(n))
            if not basis or max(basis)>2:
                raise ValueError((symbol,basis))
            result[z] = dict(symbol=symbol,basis=basis,
                title=re.search(r'TITEL\s*=\s*([^\n]+)',text).group(1).strip(),
                sha256=hashlib.sha256(raw).hexdigest(),member=name)
    return result


def training_statistics(samples, elements=None):
    mean = torch.zeros(29,WIDTH)
    scale = torch.ones_like(mean)
    for z in (ELEMENTS if elements is None else elements).values():
        values = torch.cat([s['target_aug'][s['numbers']==z] for s in samples])
        mask = torch.cat([s['aug_mask'][s['numbers']==z] for s in samples])
        if not len(values):
            raise ValueError(f'No training atoms for element {z}')
        for b in BLOCKS:
            section = slice(b['start'],b['stop'])
            if not mask[:,section].any():
                continue
            if b['l']==0:
                mean[z,section] = values[:,section].mean()
            residual = values[:,section]-mean[z,section]
            # One scale for all m in a block preserves rotation equivariance.
            scale[z,section] = residual.square().mean().sqrt().clamp_min(1e-3)
    return mean,scale


def read_form_factors(potcar_zip: Path, elements=POTCAR_ELEMENTS) -> dict[str, AtomicFormFactor]:
    factors: dict[str, AtomicFormFactor] = {}
    with zipfile.ZipFile(potcar_zip) as archive:
        for symbol in elements:
            member = f"{POTCAR_PREFIX}/{symbol}/POTCAR"
            lines = archive.read(member).decode("latin1").splitlines()
            marker = next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "atomic pseudo charge-density"
            )
            values, next_line = _floats_after(lines, marker, TABLE_SIZE)
            reciprocal_cutoff = float(lines[next_line].split()[0])
            zval_line = next(line for line in lines if "ZVAL" in line)
            zval = float(zval_line.split("ZVAL", 1)[1].split("=", 1)[1].split()[0])
            if not np.isclose(values[0], zval, atol=1e-7, rtol=0.0):
                raise ValueError(
                    f"{symbol}: form factor at G=0 is {values[0]}, ZVAL is {zval}"
                )
            factors[symbol] = AtomicFormFactor(
                symbol=symbol,
                title=lines[0].strip(),
                zval=zval,
                reciprocal_cutoff=reciprocal_cutoff,
                values=values,
                spline=CubicSpline(
                    np.arange(TABLE_SIZE, dtype=np.float64),
                    values,
                    bc_type="natural",
                    extrapolate=False,
                ),
            )
    return factors
