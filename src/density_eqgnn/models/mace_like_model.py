"""MACE-style tensor-product network for compact density coefficients.

This is intentionally small enough for the 500-molecule benchmark, but unlike
``CompactCoefficientGNN`` it carries non-scalar irreducible representations
through every interaction layer.  Repeated tensor products therefore build
many-body angular features instead of applying spherical harmonics only in the
readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn import o3
from e3nn.nn import NormActivation
from torch import nn

from density_eqgnn.periodic import atom_graph_geometry, gather_neighbors

from .model import GaussianEdgeBasis, _mlp


@dataclass
class MaceLikeModelConfig:
    num_radial: int = 12
    # ``lmax`` is retained as the backward-compatible value used by the
    # original model, where hidden and output irreps had the same bandwidth.
    lmax: int = 3
    hidden_lmax: int | None = None
    density_lmax: int | None = None
    readout_edge_lmax: int | None = None
    scalar_channels: int = 24
    vector_channels: int = 12
    tensor_channels: int = 6
    octupole_channels: int = 3
    num_interactions: int = 3
    edge_basis: int = 16
    radial_hidden: int = 64
    cutoff: float = 5.0
    num_species: int = 119
    pbc: bool = False
    learnable_atomic_baseline: bool = False

    @property
    def angular_channels(self) -> int:
        return (self.resolved_density_lmax + 1) ** 2

    @property
    def coefficient_size(self) -> int:
        return self.num_radial * self.angular_channels

    @property
    def resolved_hidden_lmax(self) -> int:
        return self.lmax if self.hidden_lmax is None else self.hidden_lmax

    @property
    def resolved_density_lmax(self) -> int:
        return self.lmax if self.density_lmax is None else self.density_lmax

    @property
    def resolved_readout_edge_lmax(self) -> int:
        if self.readout_edge_lmax is None:
            return self.resolved_hidden_lmax
        return self.readout_edge_lmax

    @property
    def maximum_readout_lmax(self) -> int:
        """Largest order reachable by the final Clebsch-Gordan product."""

        return self.resolved_hidden_lmax + self.resolved_readout_edge_lmax

    @property
    def hidden_irreps(self) -> o3.Irreps:
        multiplicities = (
            self.scalar_channels,
            self.vector_channels,
            self.tensor_channels,
            self.octupole_channels,
        )
        terms = []
        for degree in range(self.resolved_hidden_lmax + 1):
            parity = "e" if degree % 2 == 0 else "o"
            terms.append(f"{multiplicities[degree]}x{degree}{parity}")
        return o3.Irreps(" + ".join(terms))


class TensorProductInteraction(nn.Module):
    """One radial-weighted tensor-product message-passing interaction."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_edge: o3.Irreps,
        irreps_out: o3.Irreps,
        edge_basis: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_edge,
            irreps_out,
            internal_weights=False,
            shared_weights=False,
            irrep_normalization="component",
            path_normalization="element",
        )
        self.radial_weights = _mlp(
            edge_basis,
            radial_hidden,
            self.tensor_product.weight_numel,
        )
        self.skip = o3.Linear(irreps_in, irreps_out)
        self.activation = NormActivation(
            irreps_out,
            torch.nn.functional.silu,
            normalize=True,
            bias=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_harmonics: torch.Tensor,
        radial_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, _ = node_features.shape
        source = gather_neighbors(node_features, source_indices)
        weights = self.radial_weights(radial_edges)
        flat_source = source.reshape(-1, source.shape[-1])
        flat_harmonics = edge_harmonics.reshape(
            -1, edge_harmonics.shape[-1]
        )
        flat_weights = weights.reshape(-1, weights.shape[-1])
        messages = self.tensor_product(
            flat_source, flat_harmonics, flat_weights
        ).reshape(batch, atoms, pair_mask.shape[2], -1)
        messages *= pair_mask[..., None].to(messages.dtype)
        count = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=2) / torch.sqrt(
            count.to(messages.dtype)
        )
        return self.activation(self.skip(node_features) + aggregate)


class MaximumOrderReadout(nn.Module):
    """Produce high-order atom coefficients in one final tensor product.

    The message-passing trunk can remain at ``hidden_lmax`` while this readout
    couples its atom features to local edge harmonics.  If the edge harmonics
    stop at ``edge_lmax``, the Clebsch-Gordan selection rule gives the exact
    architectural ceiling ``density_lmax <= hidden_lmax + edge_lmax``.
    """

    def __init__(
        self,
        hidden_irreps: o3.Irreps,
        edge_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        edge_basis: int,
        radial_hidden: int,
    ) -> None:
        super().__init__()
        self.direct = o3.Linear(hidden_irreps, output_irreps)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            hidden_irreps,
            edge_irreps,
            output_irreps,
            internal_weights=False,
            shared_weights=False,
            irrep_normalization="component",
            path_normalization="element",
        )
        self.radial_weights = _mlp(
            edge_basis,
            radial_hidden,
            self.tensor_product.weight_numel,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_harmonics: torch.Tensor,
        radial_edges: torch.Tensor,
        pair_mask: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, _ = node_features.shape
        # Linear calibration can retain the frozen nonlinear features in their
        # original precision while evaluating ill-conditioned output weights
        # in float64. The ordinary float32 model follows the same operations.
        linear = self.radial_weights[-1]
        radial_features = self.radial_weights[:-1](radial_edges)
        weights = linear(radial_features.to(linear.weight.dtype))
        source = gather_neighbors(node_features, source_indices).to(weights.dtype)
        edge_harmonics = edge_harmonics.to(weights.dtype)
        messages = self.tensor_product(
            source.reshape(-1, source.shape[-1]),
            edge_harmonics.reshape(-1, edge_harmonics.shape[-1]),
            weights.reshape(-1, weights.shape[-1]),
        ).reshape(batch, atoms, pair_mask.shape[2], -1)
        messages *= pair_mask[..., None].to(messages.dtype)
        count = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=2) / torch.sqrt(count.to(messages.dtype))
        return self.direct(node_features.to(self.direct.weight.dtype)) + aggregate


class MaceLikeCoefficientGNN(nn.Module):
    """Predict l=0..3 density coefficients with tensor-product interactions."""

    def __init__(self, config: MaceLikeModelConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = MaceLikeModelConfig(**config)
        hidden_lmax = config.resolved_hidden_lmax
        density_lmax = config.resolved_density_lmax
        readout_edge_lmax = config.resolved_readout_edge_lmax
        if hidden_lmax < 0 or hidden_lmax > 3:
            raise ValueError("MACE-like hidden features support 0 <= hidden_lmax <= 3")
        if readout_edge_lmax < 0:
            raise ValueError("readout_edge_lmax must be non-negative")
        if density_lmax < 0:
            raise ValueError("density_lmax must be non-negative")
        if density_lmax > config.maximum_readout_lmax:
            raise ValueError(
                "density_lmax exceeds the one-step tensor-product limit: "
                f"{density_lmax} > hidden_lmax({hidden_lmax}) + "
                f"readout_edge_lmax({readout_edge_lmax})"
            )
        self.config = config
        self.input_irreps = o3.Irreps(f"{config.scalar_channels}x0e")
        self.edge_irreps = o3.Irreps.spherical_harmonics(hidden_lmax)
        self.readout_edge_irreps = o3.Irreps.spherical_harmonics(
            readout_edge_lmax
        )
        self.hidden_irreps = config.hidden_irreps
        self.output_irreps = o3.Irreps(
            " + ".join(
                f"{config.num_radial}x{degree}"
                f"{'e' if degree % 2 == 0 else 'o'}"
                for degree in range(density_lmax + 1)
            )
        )
        self.embedding = nn.Embedding(
            config.num_species,
            config.scalar_channels,
            padding_idx=0,
        )
        self.edge_basis = GaussianEdgeBasis(
            config.edge_basis, config.cutoff
        )
        interactions = []
        irreps_in = self.input_irreps
        for _ in range(config.num_interactions):
            interactions.append(
                TensorProductInteraction(
                    irreps_in,
                    self.edge_irreps,
                    self.hidden_irreps,
                    config.edge_basis,
                    config.radial_hidden,
                )
            )
            irreps_in = self.hidden_irreps
        self.interactions = nn.ModuleList(interactions)
        self.uses_maximum_order_readout = (
            density_lmax != hidden_lmax
            or config.hidden_lmax is not None
            or config.density_lmax is not None
            or config.readout_edge_lmax is not None
        )
        if self.uses_maximum_order_readout:
            self.readout = MaximumOrderReadout(
                self.hidden_irreps,
                self.readout_edge_irreps,
                self.output_irreps,
                config.edge_basis,
                config.radial_hidden,
            )
        else:
            # Preserve checkpoint compatibility for the original lmax-only
            # MACE-like model.
            self.readout = o3.Linear(self.hidden_irreps, self.output_irreps)
        self.register_buffer(
            "atomic_scalar_baseline",
            torch.zeros(config.num_species, config.num_radial),
        )
        self.learned_scalar_baseline = (
            nn.Embedding(
                config.num_species,
                config.num_radial,
                padding_idx=0,
            )
            if config.learnable_atomic_baseline
            else None
        )
        if self.learned_scalar_baseline is not None:
            nn.init.zeros_(self.learned_scalar_baseline.weight)

    def set_scalar_baseline(self, values: dict[int, list[float]]) -> None:
        baseline = torch.zeros_like(self.atomic_scalar_baseline)
        for atomic_number, coefficients in values.items():
            baseline[int(atomic_number)] = torch.as_tensor(
                coefficients,
                dtype=baseline.dtype,
                device=baseline.device,
            )
        self.atomic_scalar_baseline.copy_(baseline)
        if self.learned_scalar_baseline is not None:
            self.learned_scalar_baseline.weight.data.copy_(baseline)

    def _geometry(
        self,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return atom_graph_geometry(
            positions, atom_mask, cell, self.config.cutoff, self.config.pbc
        )

    def _degree_major_to_radial_major(
        self, degree_major: torch.Tensor
    ) -> torch.Tensor:
        blocks = degree_major.new_zeros(
            *degree_major.shape[:-1],
            self.config.num_radial,
            self.config.angular_channels,
        )
        offset = 0
        for degree in range(self.config.resolved_density_lmax + 1):
            width = 2 * degree + 1
            size = self.config.num_radial * width
            block = degree_major[..., offset : offset + size].reshape(
                *degree_major.shape[:-1],
                self.config.num_radial,
                width,
            )
            blocks[
                ..., degree**2 : (degree + 1) ** 2
            ] = block
            offset += size
        return blocks

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        distances, directions, pair_mask, source_indices = self._geometry(
            positions, atom_mask, cell
        )
        radial_edges = self.edge_basis(distances)
        edge_harmonics = o3.spherical_harmonics(
            list(range(self.config.resolved_hidden_lmax + 1)),
            directions,
            normalize=True,
            normalization="integral",
        )
        features = self.embedding(atomic_numbers)
        for interaction in self.interactions:
            features = interaction(
                features, edge_harmonics, radial_edges, pair_mask, source_indices
            )
            features *= atom_mask[..., None].to(features.dtype)
        if self.uses_maximum_order_readout:
            readout_harmonics = o3.spherical_harmonics(
                list(range(self.config.resolved_readout_edge_lmax + 1)),
                directions,
                normalize=True,
                normalization="integral",
            )
            degree_major = self.readout(
                features,
                readout_harmonics,
                radial_edges,
                pair_mask,
                source_indices,
            )
        else:
            degree_major = self.readout(features)
        blocks = self._degree_major_to_radial_major(degree_major)
        if self.learned_scalar_baseline is None:
            scalar_baseline = self.atomic_scalar_baseline[atomic_numbers]
        else:
            scalar_baseline = self.learned_scalar_baseline(atomic_numbers)
        blocks[..., 0] += scalar_baseline
        blocks *= atom_mask[..., None, None].to(blocks.dtype)
        return {
            "coefficient_blocks": blocks,
            "coefficients": blocks.reshape(
                *blocks.shape[:2], self.config.coefficient_size
            ),
        }
