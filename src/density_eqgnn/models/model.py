"""Top-level density-supervised EQGNN model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import AtomToGridDecoder
from .density_encoder import DensityToAtomEncoder
from .eqgnn import ChargeE3LikeAtomNetwork
from .heads import PropertyHead


@dataclass
class ModelConfig:
    latent_dim: int = 128
    hidden_dim: int = 256
    num_interactions: int = 4
    num_basis: int = 20
    cutoff: float = 5.0
    num_species: int = 119
    property_dim: int = 1
    positive_density: bool = True
    pbc: bool = True


class DensitySupervisedEQGNN(nn.Module):
    """Multi-task model with density teacher latent and structure latent."""

    def __init__(self, config: ModelConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        if isinstance(config, dict):
            config = ModelConfig(**config)
        self.config = config

        self.density_encoder = DensityToAtomEncoder(
            latent_dim=config.latent_dim,
            num_species=config.num_species,
            num_basis=config.num_basis,
            cutoff=config.cutoff,
            hidden_dim=config.hidden_dim,
            pbc=config.pbc,
        )
        self.structure_encoder = ChargeE3LikeAtomNetwork(
            latent_dim=config.latent_dim,
            num_interactions=config.num_interactions,
            num_species=config.num_species,
            num_basis=config.num_basis,
            cutoff=config.cutoff,
            hidden_dim=config.hidden_dim,
            pbc=config.pbc,
        )
        self.decoder = AtomToGridDecoder(
            latent_dim=config.latent_dim,
            num_basis=config.num_basis,
            cutoff=config.cutoff,
            hidden_dim=config.hidden_dim,
            positive_density=config.positive_density,
            pbc=config.pbc,
        )
        self.property_head = PropertyHead(
            latent_dim=config.latent_dim,
            out_dim=config.property_dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atomic_numbers = batch["atomic_numbers"]
        positions = batch["positions"]
        atom_mask = batch["atom_mask"]
        probe_positions = batch["probe_positions"]
        probe_mask = batch["probe_mask"]
        density = batch.get("density")
        cell = batch.get("cell")

        z_structure = self.structure_encoder(atomic_numbers, positions, atom_mask, cell)
        density_from_structure = self.decoder(
            z_structure, positions, probe_positions, atom_mask, probe_mask, cell
        )
        property_pred = self.property_head(z_structure, atom_mask)

        output = {
            "z_structure": z_structure,
            "density": density_from_structure,
            "property": property_pred,
        }

        if density is not None:
            z_teacher = self.density_encoder(
                atomic_numbers,
                positions,
                probe_positions,
                density,
                atom_mask,
                probe_mask,
                cell,
            )
            density_from_teacher = self.decoder(
                z_teacher, positions, probe_positions, atom_mask, probe_mask, cell
            )
            output["z_teacher"] = z_teacher
            output["density_teacher_recon"] = density_from_teacher

        return output

