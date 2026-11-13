"""Equivariant atom-coefficient model."""

from .compact_model import CompactCoefficientGNN, CompactModelConfig
from .mace_like_model import MaceLikeCoefficientGNN, MaceLikeModelConfig
from .model import EquivariantCoefficientGNN, ModelConfig


def build_compact_coefficient_model(config: dict):
    """Build the requested compact coefficient architecture."""

    values = dict(config)
    architecture = values.pop("architecture", "simple")
    if architecture == "simple":
        return CompactCoefficientGNN(CompactModelConfig(**values))
    if architecture == "mace_like":
        return MaceLikeCoefficientGNN(MaceLikeModelConfig(**values))
    raise ValueError(f"Unknown coefficient architecture: {architecture}")


def load_compact_model_state(model, checkpoint):
    """Honor calibrated output precision instead of silently downcasting it."""
    precision = checkpoint.get("linear_readout_dtype")
    if precision is not None:
        if precision != "float64":
            raise ValueError(f"Unsupported calibrated readout precision: {precision}")
        from density_eqgnn.frozen_readout import promote_linear_readout
        promote_linear_readout(model)
    model.load_state_dict(checkpoint["model"])
    return model

__all__ = [
    "CompactCoefficientGNN",
    "CompactModelConfig",
    "MaceLikeCoefficientGNN",
    "MaceLikeModelConfig",
    "build_compact_coefficient_model",
    "load_compact_model_state",
    "EquivariantCoefficientGNN",
    "ModelConfig",
]
