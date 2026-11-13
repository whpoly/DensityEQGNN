"""Write the final Markdown comparison for the full NMC benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


CHARGE3NET_NMC_PERCENT = 0.060
CHARGE3NET_NMC_STDERR_PERCENT = 0.001
CHARGE3NET_PAPER = "https://www.nature.com/articles/s41524-024-01343-1"
CHARGE3NET_REPOSITORY = "https://github.com/AIforGreatGood/charge3net"
CHARGE3NET_COMMIT = "18290c63c0307792ab6da5ff98ee535cdb7376d6"


def checkpoint_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(f"No model state found in {path}")
    return sum(value.numel() for value in state.values() if torch.is_tensor(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--charge3net-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    proposed_percent = 100.0 * float(metrics["density_nmae"])
    proposed_parameters = int(metrics["parameter_count"])
    charge3net_parameters = checkpoint_parameter_count(
        args.charge3net_checkpoint
    )
    ratio = charge3net_parameters / proposed_parameters
    mean_total_seconds = float(metrics["mean_coefficient_seconds"]) + float(
        metrics["mean_reconstruction_seconds"]
    )

    text = f"""# Full NMC benchmark: low-l MACE to high-l density

## Protocol

- Dataset: all 2,000 structures from the public NMC charge-density dataset.
- Official split: 1,450 train / 50 validation / 500 test.
- Proposed trunk: MACE-style equivariant message passing with hidden
  representations through l=2.
- Proposed readout: one l<=2 local tensor product, producing density
  coefficients through the Clebsch-Gordan ceiling l=4.
- Density basis: 12 radial functions and 25 real spherical-harmonic channels,
  or 300 coefficients per atom.
- Training loss: uniformly sampled real-space L1 density loss.
- Test metric: full-grid NMAE,
  sum(abs(rho_pred-rho_DFT)) / sum(abs(rho_DFT)).

## Results

| Model | Parameters | Full-grid test NMAE |
|---|---:|---:|
| low-l MACE -> l=4 coefficients | {proposed_parameters:,} | {proposed_percent:.4f}% |
| ChargE3Net (published NMC model) | {charge3net_parameters:,} | \
{CHARGE3NET_NMC_PERCENT:.3f}% +/- {CHARGE3NET_NMC_STDERR_PERCENT:.3f}% |

The coefficient model is {ratio:.2f}x smaller by parameter count. Its measured
mean end-to-end density reconstruction time is {mean_total_seconds:.3f} s per
test structure on the benchmark machine; coefficient inference itself takes
{float(metrics["mean_coefficient_seconds"]):.3f} s and the complete grid
reconstruction takes {float(metrics["mean_reconstruction_seconds"]):.3f} s.

ChargE3Net's comparison value is the published full-NMC result rather than a
sampled-probe approximation. The official checkpoint and implementation are
from commit `{CHARGE3NET_COMMIT}`.

Sources: [ChargE3Net paper]({CHARGE3NET_PAPER}),
[official implementation]({CHARGE3NET_REPOSITORY}).
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
