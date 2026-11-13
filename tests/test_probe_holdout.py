import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from scripts.nmc_probe_holdout import (DEFAULT_ROOT, PeriodicQueryGraph, atom_batch,
                                     compact_prediction, official_model, probe_batch, save_split, spatial_split)


def test_spatial_split_disjoint_stable_and_source_guard(tmp_path):
    a = spatial_split(439040, 1)
    b = spatial_split(439040, 1)
    assert {k: len(a[k]) for k in ("train", "validation", "test")} == {
        "train":351232, "validation":43904, "test":43904}
    assert np.array_equal(np.sort(np.concatenate([a[k] for k in ("train", "validation", "test")])), np.arange(439040))
    assert np.isin(a["selection"], a["validation"]).all()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["test"], spatial_split(439040, 2)["test"])
    save_split(tmp_path, 100, 1, "original")
    with pytest.raises(ValueError, match="source"):
        save_split(tmp_path, 100, 1, "different")


def sample_data():
    cell = torch.tensor([[2.9,0.,0.],[1.4,2.7,0.],[.3,.4,3.3]], dtype=torch.float64)
    positions = torch.tensor([[.1,.3,.2],[1.3,-.2,.7]], dtype=torch.float64) @ cell
    probes = torch.tensor([[.03,.2,.1],[1.2,-.1,.9],[.8,.7,.6]], dtype=torch.float64) @ cell
    return {"positions":positions, "probes":probes, "cell":cell, "numbers":torch.tensor([8,28])}


def test_periodic_query_edges_match_independent_ase_and_duplicate_queries():
    s = sample_data()
    graph = PeriodicQueryGraph(s["positions"], s["probes"], s["cell"], 4.)
    atoms = Atoms(numbers=[8,28,0,0,0], positions=torch.cat((s["positions"],s["probes"])).numpy(),
                  cell=s["cell"].numpy(), pbc=True)
    i,j,shifts = neighbor_list("ijS", atoms, 4., self_interaction=False)
    keep = (i >= 2) & (j < 2)
    expected = sorted((int(a),int(b-2),*map(int,shift)) for a,b,shift in zip(j[keep],i[keep],shifts[keep]))
    a,b,t = graph.select(torch.arange(3))
    actual = sorted((int(x),int(y),*map(int,z)) for x,y,z in zip(a,b,t))
    assert actual == expected
    a,b,t = graph.select(torch.tensor([2,0,2]))
    assert len(a[b==0]) == len(a[b==2])
    torch.testing.assert_close(a[b==0],a[b==2])
    torch.testing.assert_close(t[b==0],t[b==2])


@pytest.mark.skipif(not (DEFAULT_ROOT / "DeepDFT/densitymodel.py").exists(), reason="official checkout not downloaded")
def test_official_adapter_rotation_and_query_batch_independence():
    torch.set_num_threads(2)
    torch.manual_seed(81)
    model,_ = official_model(DEFAULT_ROOT / "DeepDFT", "cpu")
    s = sample_data()
    s["graph"] = PeriodicQueryGraph(s["positions"],s["probes"],s["cell"],4.)
    s["batch"] = atom_batch(s)
    with torch.no_grad():
        direct = model(probe_batch(s,torch.arange(3)))[0]
        rep = model.atom_model(s["batch"])
        separate = model.probe_model(probe_batch(s,torch.tensor([2,0,2])),*rep)[0]
        torch.testing.assert_close(separate,direct[torch.tensor([2,0,2])],rtol=1e-5,atol=2e-6)
        q,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        rotated = {key: value @ q if key in ("positions","probes","cell") else value for key,value in sample_data().items()}
        rotated["graph"] = PeriodicQueryGraph(rotated["positions"],rotated["probes"],rotated["cell"],4.)
        rotated["batch"] = atom_batch(rotated)
        output = model(probe_batch(rotated,torch.arange(3)))[0]
        torch.testing.assert_close(output,direct,rtol=3e-5,atol=2e-6)


def test_r16_query_cache_matches_production_values_and_gradients():
    from density_eqgnn.compact_basis import CompactBasisConfig, reconstruct_compact_density
    s = sample_data()
    basis = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                               regular_radial=True,radial_whitening=True,radial_minimum=3/32)
    s["graph"] = PeriodicQueryGraph(s["positions"],s["probes"],s["cell"],3.)
    c = torch.randn(1,2,basis.coefficient_size,dtype=torch.float64,requires_grad=True)
    ids = torch.tensor([2,0,2])
    cached = compact_prediction(s,c,ids,basis)
    direct = reconstruct_compact_density(c,s["positions"][None],s["probes"][ids][None],
        torch.ones(1,2,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool),s["cell"][None],basis)[0]
    torch.testing.assert_close(cached,direct,rtol=2e-9,atol=2e-9)
    a = torch.autograd.grad(cached.square().sum(),c)[0]
    b = torch.autograd.grad(direct.square().sum(),c)[0]
    torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-9)
