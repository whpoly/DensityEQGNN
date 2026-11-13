"""Check the NMC targets against upstream VASP I/O and exercise a native SACD/SCF pair."""
from pathlib import Path
import tempfile
from zipfile import ZipFile

import lz4.frame
import numpy as np
import torch

from scripts.nmc_eden_fullgrid import (
    DATA, PUBLIC, ROOT, NMCReader, build_model, dataset, enable_upstream, filelist,
    manifest, optimizer_for, potcar_files, save, step_loss,
)


def check_labels(tmp):
    from edengnn.data.io.vasp.parse_density import _read_chgcar, parse_aug
    reader = NMCReader(None)
    cfg = manifest("nmc")
    rows = []
    species = set()
    for index in cfg["split"]["train"]:
        entry = cfg["entries"][index]
        path = tmp / "CHGCAR"
        with Path(entry["archive"]).open("rb") as archive:
            archive.seek(entry["offset"])
            path.write_bytes(lz4.frame.decompress(archive.read(entry["size"])))
        z, cell, pos, _, density, lines = _read_chgcar(str(path))
        aug, mask = parse_aug(lines, z, reader.lmix_max, reader.basis)
        sample = torch.load(DATA / str(index) / "sample.pt", weights_only=False)
        mapped, mapped_mask = reader.occupations(sample)
        np.testing.assert_array_equal(z, sample["numbers"].numpy())
        np.testing.assert_allclose(pos, sample["positions"].numpy(), atol=2e-7, rtol=0)
        np.testing.assert_allclose(cell, sample["cell"].numpy(), atol=2e-8, rtol=0)
        target = np.load(DATA / str(index) / "rho.npy").reshape(density.shape)
        np.testing.assert_array_equal(density.astype(target.dtype), target)
        np.testing.assert_array_equal(mapped_mask, mask.astype(bool))
        np.testing.assert_array_equal(mapped, aug.astype(mapped.dtype))
        species.update(z.tolist())
        rows.append(dict(id=index, grid=list(density.shape), points=density.size,
                         valid_paw=int(mask.sum()), species=sorted(set(z.tolist()))))
        if len(rows) >= 2 and len(species) == 5:
            return rows
    raise ValueError("Training-label check did not cover all five NMC elements")


def check_native_step(tmp):
    from edengnn.data.io.vasp.parse_density import IO_VASP
    name = "al2o3_111_400"
    with ZipFile(PUBLIC / "data.zip") as published:
        for stage in ("scf", "sad"):
            file = tmp / stage / name / "CHGCAR"
            file.parent.mkdir(parents=True)
            file.write_bytes(published.read(f"data/vasp/md/al2o3/{stage}/{name}/CHGCAR"))
    pp = tmp / "pseudo.json"
    save(pp, potcar_files({"Al","O"}, "data/vasp/pseudo.json"))
    reader = IO_VASP(stage="train", dir=str(tmp / "scf"), lmix_max=2, path_pp=pp)
    listing = filelist(tmp, "native_example", [str(tmp / "sad" / name)])
    data = dataset(listing, reader)[0].cuda()
    model = build_model(150000)
    optimizer = optimizer_for(model)
    step_records = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss, density_mae, aug_mae = step_loss(model.train(), data.clone())
        loss.backward()
        norms = {}
        for key in ("representation", "probe", "aug"):
            grads = [p.grad.detach().norm() for p in getattr(model, key).parameters() if p.grad is not None]
            norm = float(torch.stack(grads).norm())
            if not 0 < norm < float("inf"):
                raise FloatingPointError(f"Invalid gradient in {key}: {norm}")
            norms[key] = norm
        optimizer.step()
        step_records.append(dict(step=step, full_grid_mae=float(density_mae), masked_paw_mae=float(aug_mae),
                                 loss=float(loss.detach()), gradient_norms=norms))
    model.eval()
    with torch.no_grad():
        first = model(data.clone())["grid_func_out"]
        model.probe.chunk_size_predict = 300000
        second = model(data.clone())["grid_func_out"]
        difference = float((first-second).abs().double().sum()/first.abs().double().sum())
        if first.shape != data.grid_func_out.shape or not np.isfinite(difference) or difference > 5e-5:
            raise ValueError(f"Memory chunking changed the full-grid result: {difference}")
    return dict(material="Al2O3", source="Published native zero-step and SCF example; diagnostic only",
                shape=data.grid_shape.tolist(), points=data.grid_func_out.numel(),
                atom_grid_pairs=int(data.npb_total), valid_paw=int(data.aug_mask.sum()),
                relative_l1_chunk_difference=difference, training=step_records)


if __name__ == "__main__":
    dll = enable_upstream()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_per_process_memory_fraction(.16)
    with tempfile.TemporaryDirectory() as work:
        tmp = Path(work)
        checked = check_labels(tmp)
        save(ROOT / "nmc_label_audit.json", checked)
        print("MATCHED_NMC_LABELS", checked, flush=True)
        result = check_native_step(tmp)
        save(ROOT / "native_fullgrid_check.json", result)
        print("FULL_GRID_CHECK", result, flush=True)
