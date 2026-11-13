"""Download a tiny non-magnetic Materials Project charge-density validation set.

The API key is read only from ``MP_API_KEY``.  It is never written to disk.
Only final CHGCAR files and non-sensitive material metadata are saved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emmet.core.summary import HasProps
from mp_api.client import MPRester


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-sites", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY in the current process environment")

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with MPRester(api_key) as mpr:
        candidates = mpr.materials.summary.search(
            has_props=[HasProps.charge_density],
            is_stable=True,
            num_sites=(1, args.max_sites),
            total_magnetization=(0.0, 1e-5),
            fields=[
                "material_id",
                "formula_pretty",
                "nsites",
                "structure",
                "total_magnetization",
            ],
            num_chunks=1,
            chunk_size=max(args.count * 5, 50),
        )
        candidates = sorted(candidates, key=lambda doc: (doc.nsites, str(doc.material_id)))
        for document in candidates:
            material_id = str(document.material_id)
            destination = args.output / material_id
            destination.mkdir(exist_ok=True)
            try:
                chgcar = mpr.get_charge_density_from_material_id(material_id)
                if chgcar is None:
                    continue
                chgcar.write_file(destination / "CHGCAR")
                document.structure.to(filename=destination / "POSCAR", fmt="poscar")
            except Exception as error:  # API objects can be absent despite summary flags.
                print(f"skip {material_id}: {error}")
                continue
            record = {
                "material_id": material_id,
                "formula": document.formula_pretty,
                "num_sites": int(document.nsites),
                "total_magnetization": float(document.total_magnetization or 0.0),
            }
            records.append(record)
            print(
                f"downloaded {len(records)}/{args.count}: "
                f"{material_id} {document.formula_pretty} ({document.nsites} sites)"
            )
            if len(records) >= args.count:
                break

    if len(records) < args.count:
        raise RuntimeError(
            f"Only downloaded {len(records)} of the requested {args.count} materials"
        )
    (args.output / "metadata.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
