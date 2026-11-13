"""Pinned official EdenGNN density classes and an audited sampled-grid adapter.

The upstream class ASTs are executed unchanged. Unused PAW / Fortran imports
are omitted so density-only experiments do not require the DFT I/O extension.
The sampled adapter preserves the upstream rounded-grid spherical stencil,
including periodic multiplicities; it does NOT replace it with an atom cutoff.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from scripts.nmc_probe_holdout import PeriodicQueryGraph, DEFAULT_ROOT, save_split, sha256
from scripts.compare_local_nmc_oracles import load_sample

ROOT = Path("runs/edengnn_nmc_2026-10-28")
REPO = ROOT / "EdenGNN"
COMMIT = "3be8c06265c02ecf1ef0587500ee9b8e091186ef"


def official_classes():
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT or subprocess.check_output(["git", "-C", str(REPO), "diff", "--name-only"], text=True).strip():
        raise ValueError("Pinned upstream checkout changed")
    for path in (ROOT / "python_deps", REPO / "src"):
        if str(path.resolve()) not in sys.path:
            sys.path.insert(0, str(path.resolve()))
    path = REPO / "src/edengnn/model/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"Encoder", "DensityLayer"}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module not in {
            "edengnn.data.io.vasp.basis", "edengnn.model.utils.cg"
        }:
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in names:
            keep.append(node)
    module = types.ModuleType("edengnn.model.density_benchmark")
    module.__package__ = "edengnn.model"
    module.__file__ = str(path.resolve())
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), module.__dict__)
    provenance = {"repository": "https://github.com/rubenlee11/EdenGNN", "commit": revision,
                  "model_source_sha256": sha256(path), "classes": sorted(names),
                  "loading": "unchanged official class ASTs; omit unrelated PAW/Fortran imports"}
    return module.Encoder, module.DensityLayer, provenance


class EdenDensity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        Encoder, DensityLayer, self.provenance = official_classes()
        cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))["model"]
        a, p = cfg["atom"], cfg["probe"]
        self.encoder = Encoder(cutoff=a["edge"]["cutoff"], l_max=a["l_max"],
            num_types=a["num_types"], n_radial_basis=a["edge"]["n_radial_basis"],
            irreps_node_features=a["conv"]["irreps_node_features"],
            num_conv_layers=a["conv"]["num_conv_layers"],
            num_radial_filter_layers=a["conv"]["num_radial_filter_layers"],
            num_radial_filter_neurons=a["conv"]["num_radial_filter_neurons"])
        self.density = DensityLayer(cutoff=p["edge"]["cutoff"], l_max=p["l_max"],
            irreps_node_attribute=self.encoder.one_hot.irreps_out["node_attrs"],
            irreps_node_features=a["conv"]["irreps_node_features"],
            n_channels=p["conv"]["n_channels"], n_radial_basis=p["edge"]["n_radial_basis"],
            n_z_embedding_dim=p["edge"]["n_z_embedding_dim"],
            n_z_embedding_neurons=p["edge"]["n_z_embedding_neurons"],
            n_z_embedding_layers=p["edge"]["n_z_embedding_layers"],
            num_radial_filter_layers=p["conv"]["num_radial_filter_layers"],
            num_radial_filter_neurons=p["conv"]["num_radial_filter_neurons"],
            chunk_size_predict=1000000, chunk_size_train=1000000)
        self.config = {"atom": a, "probe": p, "chunk_size_predict": 1000000,
                       "baseline": "POTCAR form-factor reconstructed SAD; no target fit",
                       "density_units": "electrons/angstrom^3"}

    def encode(self, data):
        data = data.clone()
        self.encoder(data)
        self.density.conv_to_output_node(data)
        return (self.density.reshape_irreps(data["node_charge"]),
                self.density.z_net(data["node_attrs"]), data)

    def sampled(self, sample, indices, representation=None):
        h, z, _ = self.encode(sample["atom_data"]) if representation is None else representation
        atoms, receiver, offset = sample["stencil"].select(indices)
        geometry = sample["stencil"]
        # Match official float32 arithmetic and its cyclic coordinate convention.
        displacement = geometry.edge_vec[offset] - geometry.pos_res[atoms]
        pe = displacement[:, [1, 2, 0]][:, None, :]
        pl = torch.linalg.vector_norm(pe, dim=-1)
        values = self.density.compute_chunk(pe, pl, z[atoms, None], h[atoms]).flatten()
        output = values.new_zeros(len(indices)).index_add(0, receiver, values)
        return output + sample["sad"][indices]

    def native(self, sample, representation=None):
        if representation is None:
            data = sample["native_data"].clone()
            self.encoder(data)
        else:
            data = representation[2].clone()
            for key in ("pos_n", "grid", "map_probe", "edge_vec_probes", "grid_shape", "nat", "npb", "npb_total"):
                data[key] = sample["native_data"][key]
        self.density(data)
        return data["charge"].flatten() + sample["sad"]


def stencil_numpy(cell, shape, positions, radius=4.):
    grid = cell / np.asarray(shape)[:, None]
    bounds = np.floor(radius*np.linalg.norm(np.linalg.inv(grid), axis=0)+1.).astype(int)
    offsets = np.stack(np.meshgrid(*[np.arange(-n,n+1) for n in bounds], indexing="ij"), -1).reshape(-1,3)
    vectors = offsets @ grid
    mask = np.linalg.norm(vectors, axis=1) <= radius
    offsets, vectors = offsets[mask], vectors[mask]
    pos_n = np.round(positions @ np.linalg.inv(cell) * np.asarray(shape)).astype(np.int64)
    return grid, offsets, vectors, pos_n


class EdenStencil:
    def __init__(self, sample, *, csr=True):
        device = sample["positions"].device
        shape = np.array(sample["metadata"]["grid_shape"])
        grid, offsets, vectors, pos_n = stencil_numpy(sample["cell"].cpu().numpy(), shape,
                                                     sample["positions"].cpu().numpy())
        self.grid = torch.tensor(grid, dtype=torch.float32, device=device)
        self.map_probe = torch.tensor(offsets, dtype=torch.long, device=device)
        self.edge_vec = torch.tensor(vectors, dtype=torch.float32, device=device)
        self.pos_n = torch.tensor(pos_n, dtype=torch.long, device=device)
        self.pos_res = sample["positions"].float() - self.pos_n.float() @ self.grid
        self.npb = len(offsets)
        self.num_atoms = len(pos_n)
        self.grid_shape = torch.tensor(shape, device=device)
        self.num_points = int(np.prod(shape))
        if csr:
            # CSR stores native atom*stencil pairs, preserving repeated periodic images.
            flat = []
            for n in pos_n:
                ids = (n[None] + offsets) % shape
                flat.append(np.ravel_multi_index(ids.T, shape).astype(np.int32))
            flat = np.concatenate(flat)
            order = np.argsort(flat, kind="stable")
            counts = np.bincount(flat, minlength=self.num_points)
            self.pairs = torch.tensor(order.astype(np.int32), device=device)
            self.offsets = torch.tensor(np.r_[0, counts.cumsum()], device=device)

    def select(self, ids):
        counts = self.offsets[ids+1] - self.offsets[ids]
        size = int(counts.sum())
        receiver = torch.repeat_interleave(torch.arange(len(ids), device=ids.device), counts, output_size=size)
        starts = counts.cumsum(0)-counts
        edge = self.offsets[ids][receiver] + torch.arange(size, device=ids.device)-starts[receiver]
        pair = self.pairs[edge].long()
        return pair//self.npb, receiver, pair%self.npb

    def native_fields(self):
        return {"pos_n":self.pos_n, "grid":self.grid, "map_probe":self.map_probe,
                "edge_vec_probes":self.edge_vec, "grid_shape":self.grid_shape,
                "nat":self.num_atoms, "npb":self.npb, "npb_total":self.npb*self.num_atoms}


def atom_data(sample):
    graph = PeriodicQueryGraph(sample["positions"], sample["positions"], sample["cell"], 4., exclude_self=True)
    source, receiver, shifts = graph.select(torch.arange(len(sample["numbers"]), device=sample["numbers"].device))
    return Data(z=sample["numbers"], pos=sample["positions"].float(), cell=sample["cell"].float(),
                edge_index=torch.stack((receiver,source)), nbr_shift=(shifts.double()@sample["cell"]).float())


def make_sad(sample, factors):
    from ase.data import chemical_symbols
    from scripts.evaluate_vasp_sad_nmc import sad_density
    cell = sample["cell"].cpu().numpy()
    symbols = np.array([chemical_symbols[n] for n in sample["numbers"].cpu().numpy()])
    frac = sample["positions"].cpu().numpy() @ np.linalg.inv(cell)
    raw = sad_density(symbols,frac,cell,tuple(sample["metadata"]["grid_shape"]),factors)
    expected = sum(factors[z].zval for z in symbols)
    if abs(raw.mean()-expected)>2e-7:
        raise ValueError("SAD electron count mismatch")
    return torch.tensor(raw.flatten()/sample["metadata"]["volume"], dtype=torch.float32, device=sample["positions"].device)


def prepare_samples(ids=range(8), device="cuda", *, csr=True):
    from scripts.evaluate_vasp_sad_nmc import read_form_factors
    factors = read_form_factors(Path("potpaw54.zip"))
    samples = []
    for sid in ids:
        start = time.perf_counter()
        sample = load_sample(Path(f"data/nmc_cache/{sid}.CHGCAR.lz4"),device)
        sample["id"] = sid
        split, metadata = save_split(DEFAULT_ROOT/"spatial_splits",len(sample["target"]),sid,sample["metadata"]["sha256"])
        sample["split"] = {k:torch.tensor(v,device=device) for k,v in split.items()}
        sample["split_metadata"] = metadata
        sample["normalizer"] = sample["target"][sample["split"]["train"]].abs().mean()
        sample["sad"] = make_sad(sample,factors)
        sample["atom_data"] = atom_data(sample)
        sample["stencil"] = EdenStencil(sample,csr=csr)
        sample["native_data"] = sample["atom_data"].clone()
        for k,v in sample["stencil"].native_fields().items():
            sample["native_data"][k] = v
        samples.append(sample)
        print(f"PREPARED sample={sid} native_pairs={sample['stencil'].npb*len(sample['numbers'])} seconds={time.perf_counter()-start:.2f}",flush=True)
    return samples


@torch.no_grad()
def evaluate(model,samples,chunk=2048,final=False):
    model.eval()
    rows=[]
    for sample in samples:
        rep=model.encode(sample["atom_data"])
        groups=("train","validation","test") if final else ("selection",)
        row={"sample_id":sample["id"]}
        for group in groups:
            sums=torch.zeros(4,dtype=torch.float64,device=sample["target"].device)
            for ids in sample["split"][group].split(chunk):
                pred=model.sampled(sample,ids,rep).double()
                target=sample["target"][ids]
                sums+=torch.stack(((pred-target).abs().sum(),target.abs().sum(),
                                   (pred-target).sum(),(-pred).clamp_min(0).sum()))
            row[group+"_nmae"]=float(sums[0]/sums[1])
            row[group+"_charge_error"]=float(sums[2]/sums[1])
            row[group+"_negative_fraction"]=float(sums[3]/sums[1])
        rows.append(row)
    return {"records":rows,**{g+"_macro_nmae":float(np.mean([r[g+"_nmae"] for r in rows])) for g in groups}}


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)
