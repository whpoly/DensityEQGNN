import torch

from density_eqgnn.models import EquivariantCoefficientGNN, ModelConfig


def test_scalar_vector_and_tensor_equivariance():
    torch.manual_seed(3)
    model = EquivariantCoefficientGNN(
        ModelConfig(
            num_radial=3,
            lmax=2,
            hidden_dim=24,
            num_interactions=2,
            edge_basis=6,
            cutoff=4.0,
            pbc=False,
        )
    ).eval()
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]
    )
    numbers = torch.tensor([[8, 1, 1]])
    mask = torch.tensor([[True, True, True]])
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_positions = positions @ rotation.T

    with torch.no_grad():
        original = model(numbers, positions, mask)
        rotated = model(numbers, rotated_positions, mask)

    assert torch.allclose(
        original["scalar_coefficients"],
        rotated["scalar_coefficients"],
        atol=2e-6,
        rtol=2e-6,
    )
    expected_vectors = original["vector_coefficients"] @ rotation.T
    assert torch.allclose(
        expected_vectors,
        rotated["vector_coefficients"],
        atol=2e-6,
        rtol=2e-6,
    )
    expected_tensors = torch.einsum(
        "pq,binqr,sr->binps",
        rotation,
        original["tensor_coefficients"],
        rotation,
    )
    assert torch.allclose(
        expected_tensors,
        rotated["tensor_coefficients"],
        atol=2e-6,
        rtol=2e-6,
    )
