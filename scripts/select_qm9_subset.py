"""Select a deterministic QM9 subset by total atom count."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True, help="QM9 parquet file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-atoms", type=int, default=12)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required for subset selection") from exc

    frame = pd.read_parquet(args.metadata, columns=["num_atoms"])
    selected = frame.index[frame["num_atoms"] <= args.max_atoms][: args.samples]
    if len(selected) < args.samples:
        raise RuntimeError(
            f"Only {len(selected)} molecules have <= {args.max_atoms} atoms"
        )
    # The public VASP archive uses one-based, six-digit QM9 member names.
    member_ids = [f"{int(index) + 1:06d}" for index in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(member_ids) + "\n", encoding="utf-8")
    print(
        f"selected {len(member_ids)} molecules with <= {args.max_atoms} atoms: "
        f"{member_ids[0]}..{member_ids[-1]}"
    )


if __name__ == "__main__":
    main()
