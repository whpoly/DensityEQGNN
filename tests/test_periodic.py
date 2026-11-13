"""Independent lattice enumeration and representation-invariance regressions."""

from dataclasses import replace
import itertools

import numpy as np
import pytest
import torch
from e3nn import o3

from density_eqgnn.basis import (
    DensityBasisConfig, atom_probe_displacements_numpy,
    reconstruct_density_numpy, reconstruct_density_torch,
)
from density_eqgnn.compact_basis import CompactBasisConfig, reconstruct_compact_density
from density_eqgnn.models import (
    CompactCoefficientGNN, CompactModelConfig,
    EquivariantCoefficientGNN, ModelConfig,
    MaceLikeCoefficientGNN, MaceLikeModelConfig,
)
from density_eqgnn.periodic import (
    PERIODIC_GEOMETRY_VERSION, atom_graph_geometry, check_periodic_checkpoint, cutoff_edges,
)


DTYPE = torch.float64
SKEW = torch.tensor([[3.0, 0., 0.], [2.7, 1.2, 0.], [0.2, 0.3, 3.5]], dtype=DTYPE)


def test_periodic_checkpoint_cannot_silently_resume_old_geometry():
    with pytest.raises(ValueError, match="new training.output_dir"):
        check_periodic_checkpoint({}, pbc=True, resume=True)
    with pytest.warns(RuntimeWarning, match="corrected model"):
        check_periodic_checkpoint({}, pbc=True)
    check_periodic_checkpoint({}, pbc=False, resume=True)
    check_periodic_checkpoint(
        {"periodic_geometry_version": PERIODIC_GEOMETRY_VERSION}, pbc=True, resume=True
    )


def brute_displacements(centers, neighbors, cell, cutoff, exclude_self=False):
    """Fixed, deliberately generous image box; no production wrapping/bounds."""
    result = []
    for i, center in enumerate(centers):
        for j, neighbor in enumerate(neighbors):
            for shift in itertools.product(range(-5, 6), repeat=3):
                if exclude_self and i == j and shift == (0, 0, 0):
                    continue
                disp = neighbor + torch.tensor(shift, dtype=DTYPE) @ cell - center
                if torch.linalg.vector_norm(disp) < cutoff:
                    result.append((i, j, disp))
    return result


def compact_reference(coefficients, atoms, probes, cell, basis):
    result = torch.zeros(len(probes), dtype=DTYPE)
    scales = torch.exp(torch.linspace(
        np.log(max(basis.cutoff / 32, .08)), np.log(basis.cutoff),
        basis.num_radial, dtype=DTYPE,
    ))
    blocks = coefficients.reshape(len(atoms), basis.num_radial, (basis.lmax + 1)**2)
    for p, a, disp in brute_displacements(probes, atoms, cell, basis.cutoff):
        r = disp.norm()
        radial = torch.exp(-.5 * (r / scales)**2) * .5 * (1 + torch.cos(torch.pi * r / basis.cutoff))
        angular = o3.spherical_harmonics(
            list(range(basis.lmax + 1)), -disp, normalize=True, normalization="integral"
        )
        result[p] += torch.einsum("nm,n,m->", blocks[a], radial, angular)
    return result


def make_model(kind):
    torch.manual_seed(31)
    if kind == "mace":
        model = MaceLikeCoefficientGNN(MaceLikeModelConfig(
            num_radial=2, hidden_lmax=2, density_lmax=4, readout_edge_lmax=2,
            scalar_channels=3, vector_channels=2, tensor_channels=2,
            num_interactions=2, edge_basis=3, radial_hidden=4, cutoff=3.2, pbc=True,
        ))
    elif kind == "compact":
        model = CompactCoefficientGNN(CompactModelConfig(
            num_radial=2, lmax=3, hidden_dim=8, num_interactions=2,
            edge_basis=3, cutoff=3.2, pbc=True,
        ))
    else:
        model = EquivariantCoefficientGNN(ModelConfig(
            num_radial=2, lmax=2, hidden_dim=8, num_interactions=2,
            edge_basis=3, cutoff=3.2, pbc=True,
        ))
    # Old models zero-initialize readouts; nonzero heads are needed to test
    # actual geometry dependence rather than trivially comparing zero fields.
    for name, module in model.named_modules():
        if "head" in name and isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=.15)
            torch.nn.init.normal_(module.bias, std=.03)
    return model.double().eval()


def supercell(atoms, cell):
    translations = [torch.tensor(s, dtype=DTYPE) @ cell
                    for s in itertools.product(range(2), range(2), range(1))]
    positions = torch.cat([atoms + shift for shift in translations])
    return positions, torch.diag(torch.tensor([2., 2., 1.], dtype=DTYPE)) @ cell


def test_skew_graph_matches_independent_all_image_enumeration():
    atoms = torch.tensor([[.04, .13, .2], [.51, .62, .7]], dtype=DTYPE) @ SKEW
    mask = torch.ones(1, 2, dtype=torch.bool)
    edges = cutoff_edges(atoms[None], atoms[None], mask, mask, SKEW[None], 3.2, True, exclude_self=True)
    expected = brute_displacements(atoms, atoms, SKEW, 3.2, True)
    actual = sorted((int(i), int(j), *np.round(d.detach().numpy(), 8))
                    for i, j, d in zip(edges.center, edges.neighbor, edges.displacement))
    reference = sorted((i, j, *np.round(d.numpy(), 8)) for i, j, d in expected)
    assert actual == reference


def test_single_atom_cell_keeps_self_images_only_inside_cutoff():
    positions = torch.zeros(1, 1, 3, dtype=DTYPE)
    mask = torch.ones(1, 1, dtype=torch.bool)
    distances, directions, valid, source = atom_graph_geometry(
        positions, mask, torch.eye(3, dtype=DTYPE)[None] * 2, 2.1, True,
    )
    assert valid.sum().item() == 6
    torch.testing.assert_close(distances[valid], torch.full((6,), 2., dtype=DTYPE))
    torch.testing.assert_close(directions[valid].sum(0), torch.zeros(3, dtype=DTYPE))
    assert torch.all(source[valid] == 0)
    assert not atom_graph_geometry(positions, mask, None, 2.1, False)[2].any()


def test_skew_nearest_image_is_not_fractional_rounding():
    atom = np.zeros((1, 3))
    probe = np.array([[.49, .49, 0.]]) @ SKEW.numpy()
    result = atom_probe_displacements_numpy(atom, probe, SKEW.numpy(), True)[0, 0]
    candidates = [probe[0] + np.array(s) @ SKEW.numpy()
                  for s in itertools.product(range(-5, 6), repeat=3)]
    expected = min(candidates, key=np.linalg.norm)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert np.linalg.norm(result) < np.linalg.norm(probe[0]) / 2


def test_compact_skew_density_matches_explicit_images_with_odd_and_high_orders():
    torch.manual_seed(12)
    atoms = torch.tensor([[.11, .23, .13], [.64, .35, .6]], dtype=DTYPE) @ SKEW
    probes = torch.tensor([[.49, .49, .02], [.9, .1, .4], [.01, .8, .12]], dtype=DTYPE) @ SKEW
    basis = CompactBasisConfig(num_radial=2, lmax=4, cutoff=3.2, pbc=True)
    coeff = torch.randn(2, basis.coefficient_size, dtype=DTYPE)
    actual = reconstruct_compact_density(
        coeff[None], atoms[None], probes[None], torch.ones(1,2,dtype=torch.bool),
        torch.ones(1,3,dtype=torch.bool), SKEW[None], basis,
    )[0]
    torch.testing.assert_close(actual, compact_reference(coeff, atoms, probes, SKEW, basis), atol=1e-11, rtol=1e-11)


def test_single_atom_density_sums_both_faces_and_is_continuous():
    basis = CompactBasisConfig(num_radial=2, lmax=1, cutoff=3., pbc=True)
    coeff = torch.tensor([[.2, -.1, .3, .2, 1., .2, -.1, .4]], dtype=DTYPE)
    atom = torch.zeros(1, 3, dtype=DTYPE)
    cell = torch.eye(3, dtype=DTYPE) * 5
    probes = torch.tensor([[2.5-1e-7, .1, .2], [2.5+1e-7, .1, .2], [-2.5, .1, .2], [2.5, .1, .2]], dtype=DTYPE)
    actual = reconstruct_compact_density(coeff[None], atom[None], probes[None],
        torch.ones(1,1,dtype=torch.bool), torch.ones(1,4,dtype=torch.bool), cell[None], basis)[0]
    torch.testing.assert_close(actual, compact_reference(coeff, atom, probes, cell, basis), atol=1e-12, rtol=1e-12)
    assert abs(actual[0] - actual[1]) < 1e-7
    torch.testing.assert_close(actual[2], actual[3], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ["mace", "compact", "cartesian"])
def test_models_and_density_are_primitive_supercell_consistent(kind):
    model = make_model(kind)
    # A single atom in a skew cell exercises nonzero self images in every layer.
    atoms = torch.tensor([[.13, .21, .32]], dtype=DTYPE) @ SKEW
    replicas, expanded = supercell(atoms, SKEW)
    probes = torch.tensor([[.4, .3, .2], [.9, .8, .7]], dtype=DTYPE) @ SKEW
    with torch.no_grad():
        coeff = model(torch.tensor([[8]]), atoms[None], torch.ones(1,1,dtype=torch.bool), SKEW[None])["coefficients"]
        repeated = model(torch.tensor([[8]*4]), replicas[None], torch.ones(1,4,dtype=torch.bool), expanded[None])["coefficients"]
    assert coeff.abs().max() > 1e-5
    torch.testing.assert_close(repeated, coeff.repeat(1,4,1), atol=1e-10, rtol=1e-10)
    if kind == "cartesian":
        basis = DensityBasisConfig(num_radial=2, lmax=2, cutoff=3.2, pbc=True)
        reconstruct = reconstruct_density_torch
    else:
        basis = CompactBasisConfig(num_radial=2, lmax=4 if kind == "mace" else 3, cutoff=3.2, pbc=True)
        reconstruct = reconstruct_compact_density
    rho = reconstruct(coeff, atoms[None], probes[None], torch.ones(1,1,dtype=torch.bool), torch.ones(1,2,dtype=torch.bool), SKEW[None], basis)
    rho_repeated = reconstruct(repeated, replicas[None], probes[None], torch.ones(1,4,dtype=torch.bool), torch.ones(1,2,dtype=torch.bool), expanded[None], basis)
    torch.testing.assert_close(rho, rho_repeated, atol=1e-10, rtol=1e-10)


def test_legacy_numpy_torch_and_explicit_image_sum_agree():
    torch.manual_seed(5)
    basis = DensityBasisConfig(num_radial=2, lmax=2, cutoff=3.2, pbc=True)
    atoms = torch.tensor([[.1, .2, .3]], dtype=DTYPE) @ SKEW
    probes = torch.tensor([[.6, .5, .4], [.9, .8, .2]], dtype=DTYPE) @ SKEW
    coeff = torch.randn(1, basis.coefficient_size, dtype=DTYPE)
    value = reconstruct_density_numpy(coeff.numpy(), atoms.numpy(), probes.numpy(), SKEW.numpy(), basis)
    expected = np.zeros(2)
    for shift in itertools.product(range(-5,6), repeat=3):
        expected += reconstruct_density_numpy(coeff.numpy(), atoms.numpy()+np.array(shift)@SKEW.numpy(), probes.numpy(), None, replace(basis,pbc=False))
    np.testing.assert_allclose(value, expected, atol=1e-11, rtol=1e-11)
    actual = reconstruct_density_torch(coeff[None],atoms[None],probes[None],torch.ones(1,1,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),SKEW[None],basis)
    np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("compact", [True, False])
def test_periodic_and_nonperiodic_decoders_agree_when_no_extra_images_fit(compact):
    basis = (CompactBasisConfig(num_radial=2, lmax=4, cutoff=3., pbc=True)
             if compact else DensityBasisConfig(num_radial=2, lmax=2, cutoff=3., pbc=True))
    reconstruct = reconstruct_compact_density if compact else reconstruct_density_torch
    coeff = torch.randn(1, 2, basis.coefficient_size, dtype=DTYPE)
    atoms = torch.tensor([[[.2, .3, .4], [1.1, .7, .6]]], dtype=DTYPE)
    probes = torch.tensor([[[.8, .1, .3], [1.2, .6, .4]]], dtype=DTYPE)
    mask = torch.ones(1, 2, dtype=torch.bool)
    periodic = reconstruct(coeff, atoms, probes, mask, mask, torch.eye(3, dtype=DTYPE)[None]*20, basis)
    isolated = reconstruct(coeff, atoms, probes, mask, mask, None, replace(basis, pbc=False))
    torch.testing.assert_close(periodic, isolated, atol=1e-12, rtol=1e-12)


def test_batched_masks_unwrapped_coordinates_and_rotations():
    model = make_model("mace")
    atoms = torch.tensor([[.1, .2, .3], [.6, .4, .7]], dtype=DTYPE) @ SKEW
    probes = torch.tensor([[.4, .5, .6], [.9, .2, .1]], dtype=DTYPE) @ SKEW
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
    rotation[:, 0] *= torch.linalg.det(rotation)
    # Independent lattice translations of both atoms and queries must not
    # change the physical field, even when their coordinates are far outside.
    moved = atoms + torch.tensor([[3., -2., 1.], [-2., 4., -1.]], dtype=DTYPE) @ SKEW
    moved_probes = probes + torch.tensor([[-3., 2., -1.], [4., -2., 1.]], dtype=DTYPE) @ SKEW
    cells = torch.stack([SKEW, SKEW @ rotation.T])
    positions = torch.stack([torch.cat([atoms, atoms[:1]+100]), torch.cat([moved @ rotation.T, atoms[:1]-100])])
    queries = torch.stack([torch.cat([probes, probes[:1]+200]), torch.cat([moved_probes @ rotation.T, probes[:1]-200])])
    mask = torch.tensor([[True, True, False]] * 2)
    numbers = torch.tensor([[8, 1, 0]] * 2)
    coeff = model(numbers, positions, mask, cells)["coefficients"]
    basis = CompactBasisConfig(num_radial=2, lmax=4, cutoff=3.2, pbc=True)
    # A nonzero padded coefficient must still be ignored by the decoder.
    coeff = coeff.clone()
    coeff[:, 2] = 17
    density = reconstruct_compact_density(coeff, positions, queries, mask, mask, cells, basis)
    torch.testing.assert_close(density[0], density[1], atol=1e-10, rtol=1e-10)
    assert torch.all(density[:, 2] == 0)
    single_coeff = model(numbers[:1, :2], atoms[None], mask[:1, :2], SKEW[None])["coefficients"]
    torch.testing.assert_close(single_coeff, coeff[:1, :2], atol=1e-11, rtol=1e-11)


def test_periodic_decoder_position_cell_and_coefficient_gradients():
    basis = CompactBasisConfig(num_radial=2, lmax=1, cutoff=2.8, pbc=True)
    coeff = torch.randn(1, 1, 8, dtype=DTYPE, requires_grad=True)
    atoms = torch.tensor([[[.2, .3, .4]]], dtype=DTYPE, requires_grad=True)
    probes = torch.tensor([[[1.1, .7, .9]]], dtype=DTYPE, requires_grad=True)
    cell = SKEW[None].clone().requires_grad_()
    mask = torch.ones(1, 1, dtype=torch.bool)
    def function(c, a, p, lattice):
        return reconstruct_compact_density(c, a, p, mask, mask, lattice, basis)
    assert torch.autograd.gradcheck(function, (coeff, atoms, probes, cell), fast_mode=True)


def test_periodic_model_density_loss_backpropagates_and_empty_neighborhood_works():
    model = make_model("mace")
    positions = torch.tensor([[[.2, .3, .4], [1.1, .9, .6]]], dtype=DTYPE, requires_grad=True)
    cell = SKEW[None].clone().requires_grad_()
    coeff = model(torch.tensor([[8,1]]), positions, torch.ones(1,2,dtype=torch.bool), cell)["coefficients"]
    probes = torch.tensor([[[.4,.7,.8], [1.2,.8,.5]]], dtype=DTYPE)
    density = reconstruct_compact_density(coeff, positions, probes, torch.ones(1,2,dtype=torch.bool), torch.ones(1,2,dtype=torch.bool), cell, CompactBasisConfig(num_radial=2,lmax=4,cutoff=3.2,pbc=True))
    density.square().sum().backward()
    assert torch.isfinite(positions.grad).all() and positions.grad.norm() > 0
    assert torch.isfinite(cell.grad).all() and cell.grad.norm() > 0
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    # No periodic image within support: finite zero field with zero gradients.
    c = torch.ones(1,1,2,dtype=DTYPE,requires_grad=True)
    empty = reconstruct_compact_density(c, torch.zeros(1,1,3,dtype=DTYPE), torch.ones(1,1,3,dtype=DTYPE)*5, torch.ones(1,1,dtype=torch.bool), torch.ones(1,1,dtype=torch.bool), torch.eye(3,dtype=DTYPE)[None]*10, CompactBasisConfig(num_radial=2,lmax=0,cutoff=.1,pbc=True))
    assert empty.item() == 0
    empty.sum().backward()
    assert torch.all(c.grad == 0)
