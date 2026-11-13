"""Extract per-atom density coefficients from one or more VASP CHGCAR files."""

from __future__ import annotations

import argparse
from pathlib import Path

from density_eqgnn.basis import DensityBasisConfig
from density_eqgnn.chgcar import extract_chgcar_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CHGCAR file or directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-radial", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument("--lmax", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--fit-probes", type=int, default=20000)
    parser.add_argument("--eval-probes", type=int, default=4096)
    parser.add_argument(
        "--fit-mode",
        choices=["global", "blockwise", "stockholder"],
        default="global",
    )
    args = parser.parse_args()

    source = Path(args.input)
    files = [source] if source.is_file() else sorted(source.rglob("CHGCAR"))
    if not files:
        raise FileNotFoundError(f"No CHGCAR files found under {source}")
    output = Path(args.output)
    config = DensityBasisConfig(
        num_radial=args.num_radial,
        cutoff=args.cutoff,
        lmax=args.lmax,
        pbc=True,
    )
    for path in files:
        name = path.parent.name if path.name == "CHGCAR" else path.stem
        destination = output / f"{name}.npz" if len(files) > 1 or output.suffix != ".npz" else output
        metrics = extract_chgcar_sample(
            path,
            destination,
            config,
            ridge=args.ridge,
            fit_probes=args.fit_probes,
            eval_probes=args.eval_probes,
            fit_mode=args.fit_mode,
        )
        print(
            f"{path} -> {destination}: fit NMAE={metrics['nmae']:.6f}, "
            f"charge error={metrics['charge_error']:.3e}"
        )


if __name__ == "__main__":
    main()
