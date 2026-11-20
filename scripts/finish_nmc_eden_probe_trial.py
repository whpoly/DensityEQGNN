"""Capture the live matched control and adopt the user's validation decision once."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

import psutil
import torch
from filelock import FileLock

from scripts import nmc_eden_probe_trial as trial
from scripts import nmc_eden_fullgrid as full


MODULE = "scripts.nmc_eden_probe_trial"
EDEN_COMMAND = ["-X", "utf8", "-u", "-m", "scripts.nmc_eden_fullgrid", "--potcar-sacd"]


def shareable_snapshot(source, dest):
    # The live trainer atomically replaces last.pt. Allow that Windows rename while
    # streaming the old inode into a private file.
    if os.name == "nt":
        import msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        open_file = kernel.CreateFileW
        open_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        open_file.restype = wintypes.HANDLE
        handle = open_file(str(source.resolve()), 0x80000000, 7, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        input_file = os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb")
    else:
        input_file = source.open("rb")
    with input_file as file, dest.open("wb") as out:
        shutil.copyfileobj(file, out, length=1024 * 1024)


def capture_control():
    origin = full.read(trial.TRIAL / "origin.json")
    target = origin["origin_step"] + origin["trial_structures"]
    path = trial.TRIAL / "control.pt"
    last = trial.FULL / "last.pt"
    tmp = path.with_suffix(".snapshot")
    last_seen = None
    print("Waiting for original full-grid step", target, flush=True)
    while not path.exists():
        step = full.read(trial.FULL / "progress.json")["step"]
        modified = last.stat().st_mtime_ns
        if step >= target - 40 and modified != last_seen:
            shareable_snapshot(last, tmp)
            info = torch.load(tmp, map_location="cpu", weights_only=False)
            if info["lock"] != full.read(trial.FULL / "settings.json"):
                raise ValueError("Wrong locked settings in control checkpoint")
            captured = info["step"]
            del info
            if captured == target:
                tmp.replace(path)
                break
            tmp.unlink()
            if captured > target:
                raise RuntimeError(f"Live step {target} was missed. Replay exactly from {trial.TRIAL/'origin.pt'}.")
            last_seen = modified
        time.sleep(1 if step >= target-60 else 8)
    check = torch.load(path, map_location="cpu", weights_only=False)
    if check["step"] != target or check["lock"] != full.read(trial.FULL / "settings.json"):
        raise ValueError("Saved control does not match the planned comparison")
    return target


def matches(process, args):
    try:
        command = process.cmdline()
        return (command[1:] == args and os.path.samefile(process.exe(), sys.executable)
                and os.path.samefile(process.cwd(), Path.cwd()))
    except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
        return False


def candidate_processes():
    args = ["-X", "utf8", "-u", "-m", MODULE, "candidate"]
    return [p for p in psutil.process_iter(["name"]) if str(p.info["name"]).lower() == "python.exe" and matches(p, args)]


def restart_candidate():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    old = full.read(trial.TRIAL / "launch.json")
    args = ["-X", "utf8", "-u", "-m", MODULE, "candidate"]
    out, err = [trial.TRIAL / f"candidate-{stamp}.{suffix}" for suffix in ("log", "err")]
    with out.open("w") as log, err.open("w") as errors:
        child = subprocess.Popen([sys.executable, *args], stdout=log, stderr=errors,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    full.save(trial.TRIAL/f"launch-before-{stamp}.json", old)
    full.save(trial.TRIAL/"launch.json", dict(old, pid=child.pid, previous_pid=old["pid"], stdout=str(out.resolve()),
                                              stderr=str(err.resolve()), started=datetime.now().astimezone().isoformat()))
    return child.pid


def main():
    folder = trial.TRIAL
    with FileLock(folder/"coordinator.lock", timeout=0):
        if (folder/"superseded.json").exists() or (folder/"handoff.json").exists():
            return
        target = capture_control()
        if not (folder/"control_validation.json").exists():
            subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", MODULE, "control"], check=True)
        # A future decision must not race an exiting trial worker at its 725th update.
        if not (folder/"decision.json").exists():
            while not (folder/"sampled_validation.json").exists() or full.read(folder/"progress.json")["phase"] != "awaiting_comparison":
                if not candidate_processes():
                    raise RuntimeError("Candidate stopped before reaching its comparison gate")
                time.sleep(3)
            while candidate_processes():
                time.sleep(1)
            subprocess.run([sys.executable, "-X", "utf8", "-u", "-m", MODULE, "decide"], check=True)
        decision = full.read(folder/"decision.json")
        if not decision["adopt_probes"]:
            full.save(folder/"handoff.json", dict(active_run=str(trial.FULL), decision="full_grid", control_step=target))
            return

        existing = candidate_processes()
        pid = existing[0].pid if existing else restart_candidate()
        end = time.monotonic() + 240
        while full.read(folder/"progress.json").get("sampled_updates", 0) <= 725:
            if not psutil.pid_exists(pid) or time.monotonic() > end:
                raise RuntimeError("Sampled production training has not advanced past the gate; full-grid training is still live")
            time.sleep(2)
        # Retire only the matching original worker, after its updated checkpoint is
        # complete and the selected sampled worker is already stepping.
        launch = full.read(trial.FULL/"launch.json")
        retired = None
        if psutil.pid_exists(launch["pid"]):
            process = psutil.Process(launch["pid"])
            if not matches(process, EDEN_COMMAND):
                raise RuntimeError("Full-grid process identity differs from launch.json")
            process.terminate()
            process.wait(timeout=30)
            retired = process.pid
        full.save(folder/"handoff.json", dict(active_run=str(folder), decision="probes", control_step=target,
                                              sampled_pid=pid, retired_full_grid_pid=retired))
        print("ADOPTED", pid, flush=True)


if __name__ == "__main__":
    main()
