"""Convert a directory of VASP CHGCAR files into NPZ density samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from density_eqgnn.data.chgcar import chgcar_to_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing CHGCAR files.")
    parser.add_argument("--output", required=True, help="Output directory for .npz samples.")
    parser.add_argument("--pattern", default="CHGCAR", help="File name or glob pattern.")
    parser.add_argument("--probes", type=int, default=None, help="Random probes per sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--density-normalization",
        default="vasp_divide_volume",
        choices=["raw", "vasp_divide_volume"],
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {input_dir}")

    split_lines = []
    for idx, path in enumerate(files):
        name = path.relative_to(input_dir).as_posix().replace("/", "__")
        out = output_dir / f"{name}.npz"
        chgcar_to_npz(
            path,
            out,
            num_probes=args.probes,
            seed=args.seed + idx,
            density_normalization=args.density_normalization,
        )
        split_lines.append(out.name)
        print(f"wrote {out}")
    (output_dir / "all.txt").write_text("\n".join(split_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

