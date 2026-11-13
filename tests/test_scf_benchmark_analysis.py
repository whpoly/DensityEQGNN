import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_nmc_scf_benchmark.py"
SPEC = importlib.util.spec_from_file_location("analyze_nmc_scf_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
paired_summary = MODULE.paired_summary
parse_case = MODULE.parse_case


def test_parse_case_counts_steps_and_timing(tmp_path):
    (tmp_path / "OSZICAR").write_text(
        " DAV:   1   -1.0\n DAV:   2   -1.1\n DAV:   3   -1.2\n",
        encoding="utf-8",
    )
    (tmp_path / "OUTCAR").write_text(
        "aborting loop because EDIFF is reached\n"
        " free  energy   TOTEN  =       -123.456 eV\n"
        " General timing and accounting informations for this job:\n"
        " Elapsed time (sec):      12.5\n",
        encoding="utf-8",
    )

    result = parse_case(tmp_path)

    assert result["completed"]
    assert result["ediff_reached"]
    assert result["electronic_steps"] == 3
    assert result["final_toten_ev"] == -123.456
    assert result["vasp_elapsed_seconds"] == 12.5


def test_paired_summary_requires_converged_pairs():
    records = [
        {
            "sample_id": 1,
            "case": "sad",
            "ediff_reached": True,
            "electronic_steps": 10,
            "final_toten_ev": -5.0,
            "vasp_elapsed_seconds": 20.0,
        },
        {
            "sample_id": 1,
            "case": "ml",
            "ediff_reached": True,
            "electronic_steps": 6,
            "final_toten_ev": -5.000001,
            "vasp_elapsed_seconds": 13.0,
            "density_generation_seconds": 0.5,
        },
        {
            "sample_id": 2,
            "case": "sad",
            "ediff_reached": True,
            "electronic_steps": 8,
            "final_toten_ev": -4.0,
            "vasp_elapsed_seconds": 16.0,
        },
        {
            "sample_id": 2,
            "case": "ml",
            "ediff_reached": False,
            "electronic_steps": 180,
            "final_toten_ev": None,
            "vasp_elapsed_seconds": None,
            "density_generation_seconds": 0.5,
        },
    ]

    result = paired_summary(records, "ml")

    assert result["num_converged_pairs"] == 1
    assert result["mean_steps_saved"] == 4.0
    assert result["mean_relative_reduction"] == 0.4
    assert result["mean_elapsed_seconds_saved"] == 7.0
    assert result["mean_net_seconds_saved_including_density_generation"] == 6.5
    assert result["max_absolute_energy_delta_ev"] < 2e-6
