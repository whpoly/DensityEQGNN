"""User-requested 500-structure test of one frozen validation-selected snapshot."""
import argparse
import datetime
import io
import math
from pathlib import Path
import time

import numpy as np
import torch
from filelock import FileLock

from scripts import nmc_eden_early_stopping as trainer
from scripts import compare_nmc_joint_test as paired


full = trainer.full
RUN = trainer.RUN
TEST = RUN / "interim_test_20261118"
REPORT = Path("reports/nmc_eden_fullgrid/interim_test_20261118.json")


def evaluation_settings():
    paths = [Path(__file__), Path(full.__file__), Path(paired.__file__),
             Path("scripts/benchmark_public_density.py"), Path("scripts/train_deepdft_paw.py")]
    return dict(scope="user_requested_interim_test_while_training_continues",
                code_sha256={str(p): full.sha256(p) for p in paths},
                density="all 439040 original grid points per structure",
                paw="all valid coefficients", aggregation="equal weight per structure",
                density_nmae="sum(abs(pred-target))/sum(abs(target))",
                paper_density_relative_l1="sum(abs(pred-target))*volume/grid_points/reference_electrons",
                torch_threads=2, gpu_memory_fraction=0.16, tf32=False,
                used_for_training_or_stopping=False)


def freeze(cfg):
    """Read a single atomic checkpoint image, then select without test access."""
    snapshot = TEST / "snapshot.pt"
    if not snapshot.exists():
        raw = (RUN / "last.pt").read_bytes()
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if state["lock"] != full.read(RUN / "settings.json") or state["lock"] != trainer.settings(cfg):
            raise ValueError("Live checkpoint and locked training recipe differ")
        temp = snapshot.with_suffix(".tmp")
        temp.write_bytes(raw)
        temp.replace(snapshot)
        del raw
    else:
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
    lock, history = state["lock"], state["history"]
    if lock != trainer.settings(cfg) or not history or state["best_model"] is None:
        raise ValueError("Snapshot has no valid validation-selected weights")
    chosen = min(history, key=lambda row: row["density_mae"])
    if history[-1]["step"] > state["step"] or state.get("stop_reason"):
        raise ValueError("Expected an intermediate training snapshot")
    expected = dict(reason="user_requested_interim_test_without_stopping_training",
                    step=state["step"], validation=chosen, lock=lock,
                    selection_rule="minimum_validation_density_mae_in_frozen_history",
                    source_sha256=full.sha256(snapshot))
    settings_path, marker = TEST / "evaluation_settings.json", TEST / "selection.json"
    if marker.exists():
        selection = full.read(marker)
        if (selection != dict(expected, checkpoint_sha256=full.sha256(TEST / "best.pt"))
                or full.read(settings_path) != evaluation_settings()
                or full.read(TEST / "validation.json") != history):
            raise ValueError("Frozen selection, validation history or evaluator changed")
    else:
        full.save(TEST / "settings.json", lock)
        full.save(TEST / "validation.json", history)
        full.save(settings_path, evaluation_settings())
        full.pt_save(TEST / "best.pt", dict(model=state["best_model"], step=chosen["step"], lock=lock))
        selection = dict(expected, checkpoint_sha256=full.sha256(TEST / "best.pt"))
        full.save(marker, selection)
    paired.verify_selection(TEST / "best.pt", selection, history, "density_mae", lock)
    return selection, lock


def verify_deep(lock):
    frozen = full.read(paired.DEEP_TEST / "frozen.json")
    summary = full.read(paired.DEEP_SUMMARY)
    progress = full.read(paired.DEEP / "progress.json")
    if (summary["test_count"] != 500 or summary["selections"] != frozen
            or progress["step"] != paired.DEEP_LIMIT
            or trainer.deep.lock_for("nmc") != lock["deep_protocol"]):
        raise ValueError("DeepDFT provenance differs from the locked comparison")
    paired.verify_selection(paired.DEEP / "best.pt", frozen["deep_42"],
                            full.read(paired.DEEP / "history.json"), "density_mae", lock["deep_protocol"])
    path = TEST / "deep_selection.json"
    if path.exists() and full.read(path) != frozen:
        raise ValueError("DeepDFT reference weights changed")
    full.save(path, frozen)
    return frozen, summary


def progress(count, phase, selection, **extra):
    full.save(TEST / "progress.json", dict(phase=phase, completed=count, total=500,
              selected_step=selection["validation"]["step"], snapshot_step=selection["step"],
              updated=time.time(), training_continues=True, **extra))


def paired_row(record, index, selection, deep_selection):
    deep_record = full.read(paired.DEEP_TEST / "test_records" / f"{index}.json")
    if record["deep_checkpoint_sha256"] != deep_selection["deep_42"]["checkpoint_sha256"]:
        raise ValueError("Per-structure DeepDFT provenance differs")
    row = paired.paired_record(record, deep_record, index, selection["checkpoint_sha256"])
    volume_element, electrons = record["volume_element"], record["reference_electrons"]
    if not (math.isfinite(volume_element) and volume_element > 0 and math.isfinite(electrons) and electrons > 0):
        raise ValueError("Invalid density normalization")
    native_relative = record["density"]["abs_error_sum"] * volume_element / electrons
    paired.require_close(native_relative, record["density"]["paper_relative_l1"], "Paper density normalization")
    row["edengnn"]["paper_density_relative_l1_pct"] = 100 * native_relative
    row["deepdft_paw"]["paper_density_relative_l1_pct"] = (
        100 * deep_record["scores"]["deep_42"]["density"]["abs_error_sum"] * volume_element / electrons)
    return row


@torch.no_grad()
def test(cfg, selection, lock, deep_selection):
    ids = cfg["split"]["test"]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected the official 500 test structures")
    prepared = full.read(full.DATA.parent / "test_prepared.json")
    if prepared["count"] != 500 or prepared["split_sha256"] != cfg["split_sha256"]:
        raise ValueError("Prepared labels do not match the official test split")
    if {p.stem for p in (paired.DEEP_TEST / "test_records").glob("*.json")} != {str(i) for i in ids}:
        raise ValueError("DeepDFT test coverage is incomplete")
    state = torch.load(TEST / "best.pt", map_location="cpu", weights_only=False)
    if state["lock"] != lock or state["step"] != selection["validation"]["step"]:
        raise ValueError("Frozen checkpoint metadata changed")
    model = full.build_model(lock["model"]["chunk_size_predict"]).eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    del state
    reader = full.NMCReader(Path(lock["sacd_directory"]), reconstructed=True)
    examples = full.dataset(full.filelist(TEST, "test", ids), reader)
    started = time.perf_counter()
    records = []
    progress(0, "testing", selection)
    for j, index in enumerate(ids):
        path = TEST / "test_records" / f"{index}.json"
        if path.exists():
            record = full.read(path)
        else:
            tic = time.perf_counter()
            data = examples[j].cuda()
            if int(data.name) != index:
                raise ValueError("Test structure order changed")
            out = model(data)
            target = torch.from_numpy(np.load(full.DATA / str(index) / "rho.npy")).cuda().double()
            pred = out["grid_func_out"].flatten()
            if (pred.shape != target.shape or target.numel() != 439040
                    or not torch.isfinite(pred).all() or not torch.isfinite(out["aug_tensor"]).all()):
                raise ValueError(f"Invalid full-grid prediction for {index}")
            volume, electrons = float(data.volume), float(data.nelec)
            density = full.metric(pred, target, {"metadata": {"volume": volume}})
            density["paper_relative_l1"] = density["abs_error_sum"] * volume / target.numel() / electrons
            record = dict(id=index, points=target.numel(), checkpoint_sha256=selection["checkpoint_sha256"],
                          deep_checkpoint_sha256=deep_selection["deep_42"]["checkpoint_sha256"],
                          density=density, reference_electrons=electrons, volume_element=volume / target.numel(),
                          reference_integral=float(target.sum()) * volume / target.numel(),
                          occupation=full.measures(out["aug_tensor"].flatten(), data.aug_tensor.flatten(), data.aug_mask),
                          seconds=time.perf_counter() - tic)
            paired_row(record, index, selection, deep_selection)
            full.save(path, record)
            del data, out, pred, target
        paired_row(record, index, selection, deep_selection)
        records.append(record)
        progress(j + 1, "testing", selection, elapsed_seconds=time.perf_counter() - started, last_id=index)
        print("TEST", j + 1, "/", len(ids), "id", index, "seconds", record["seconds"], flush=True)
    return records


def summarize(cfg, selection, lock, deep_selection, deep_summary, records):
    ids = cfg["split"]["test"]
    if ([r["id"] for r in records] != ids
            or {p.stem for p in (TEST / "test_records").glob("*.json")} != {str(i) for i in ids}):
        raise ValueError("Incomplete or duplicate official test results")
    paired.verify_selection(TEST / "best.pt", selection, full.read(TEST / "validation.json"), "density_mae", lock)
    rows = [paired_row(r, i, selection, deep_selection) for r, i in zip(records, ids)]
    metrics = paired.paired_metrics(rows)
    draws = np.random.default_rng(full.SEED).integers(500, size=(10000, 500), dtype=np.int32)
    native = paired.paired_summary(rows, "paper_density_relative_l1_pct", draws)
    expected = deep_summary["summary"]["deep_42"]
    for key in ("density_nmae", "occupation_relative_l1"):
        paired.require_close(metrics[key]["deepdft_paw_pct"], 100 * expected[key], f"DeepDFT {key}")
    paired.require_close(metrics["occupation_mae"]["deepdft_paw"], expected["occupation_mae"], "DeepDFT PAW MAE")
    result = dict(completed=True, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope="user_requested_interim_test_while_training_continues", test_count=500,
                  split_sha256=cfg["split_sha256"], seed=42, equal_training_steps=False,
                  training_continues=True, used_for_training_or_stopping=False,
                  edengnn=dict(snapshot_step=selection["step"], selected_step=selection["validation"]["step"],
                               full_grid_training_steps=0, checkpoint_sha256=selection["checkpoint_sha256"]),
                  deepdft_paw=dict(training_steps=paired.DEEP_LIMIT,
                                   selected_step=deep_selection["deep_42"]["validation"]["step"],
                                   checkpoint_sha256=deep_selection["deep_42"]["checkpoint_sha256"]),
                  edengnn_selection=selection, deepdft_paw_selection=deep_selection["deep_42"],
                  evaluation_settings=full.read(TEST / "evaluation_settings.json"),
                  paper_density_relative_l1=native,
                  paper_reference=dict(density_relative_l1_pct=0.096, occupation_mae=0.00085,
                                       occupation_relative_l1=None, test_count=500,
                                       source="Supplementary Discussions 1, printed page 5, PDF page 6",
                                       pdf_sha256=full.sha256(full.ROOT / "supplement.pdf"),
                                       same_dft_labels=False,
                                       interpretation="Numerical reference only: authors recomputed the DFT labels",
                                       edengnn_density_relative_difference_pct=100 * (native["edengnn_pct"] / 0.096 - 1),
                                       edengnn_paw_mae_relative_difference_pct=100 * (metrics["occupation_mae"]["edengnn"] / 0.00085 - 1)),
                  records=rows, **metrics)
    full.save(REPORT, result)
    full.save(TEST / "benchmark.json", result)
    progress(500, "complete", selection, report=str(REPORT))
    print("INTERIM TEST COMPLETE", REPORT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    TEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    dll = full.enable_upstream()
    with FileLock(TEST / "test.lock", timeout=0):
        cfg = full.manifest("nmc")
        selection, lock = freeze(cfg)
        print("FROZEN", selection["step"], "selected", selection["validation"]["step"], selection["checkpoint_sha256"], flush=True)
        if not args.freeze_only:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(0.16)
            deep_selection, deep_summary = verify_deep(lock)
            try:
                records = test(cfg, selection, lock, deep_selection)
                summarize(cfg, selection, lock, deep_selection, deep_summary, records)
            except Exception as exc:
                count = len(list((TEST / "test_records").glob("*.json")))
                progress(count, "failed", selection, error=repr(exc))
                raise
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
