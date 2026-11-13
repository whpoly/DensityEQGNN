"""Download the four QM9/VASP tar shards needed for a 3,500-sample benchmark.

The DTU Figshare endpoint is slow per connection.  This downloader therefore
uses verified HTTP Range chunks, can resume after interruption, and validates
the final files against the checksums published by Figshare.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Archive:
    name: str
    file_id: int
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"https://ndownloader.figshare.com/files/{self.file_id}"


ARCHIVES = (
    Archive("000xxx.tar", 31062047, 5_526_958_080, "00f306464d76c206ce00781415d405ca"),
    Archive("001xxx.tar", 31062152, 7_160_862_720, "6eccc471ec61a1afb77eba7366e43a53"),
    Archive("002xxx.tar", 31062214, 7_252_449_280, "7658f0f2e6f5cc2c80236293ca080cde"),
    Archive("003xxx.tar", 31062266, 6_380_503_040, "e4a32c83efccdd120323700d442d27f1"),
)


def _fetch_range(url: str, start: int, stop: int, retries: int = 8) -> bytes:
    expected = stop - start + 1
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{stop}",
                    "User-Agent": "density-eqgnn-qm9-benchmark/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if len(payload) != expected:
                raise IOError(f"expected {expected} bytes, received {len(payload)}")
            return payload
        except Exception as exc:  # network retry path
            error = exc
            time.sleep(min(2**attempt, 30))
    raise IOError(f"failed range {start}-{stop} after {retries} attempts") from error


def md5sum(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download_archive(
    archive: Archive,
    output_dir: Path,
    workers: int,
    chunk_size: int,
    verify: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / archive.name
    done_dir = output_dir / f".{archive.name}.chunks"
    done_dir.mkdir(exist_ok=True)
    chunks = math.ceil(archive.size / chunk_size)

    if destination.exists() and destination.stat().st_size == archive.size:
        if not verify or md5sum(destination) == archive.md5:
            print(f"reuse verified {destination}", flush=True)
            shutil.rmtree(done_dir, ignore_errors=True)
            return destination

    with destination.open("ab") as handle:
        handle.truncate(archive.size)

    write_lock = threading.Lock()
    completed_bytes = sum(
        min(chunk_size, archive.size - int(marker.stem) * chunk_size)
        for marker in done_dir.glob("*.done")
        if marker.stem.isdigit() and int(marker.stem) < chunks
    )
    started = time.monotonic()

    def fetch(index: int) -> tuple[int, int]:
        marker = done_dir / f"{index}.done"
        if marker.exists():
            return index, 0
        start = index * chunk_size
        stop = min(archive.size, start + chunk_size) - 1
        payload = _fetch_range(archive.url, start, stop)
        with destination.open("r+b", buffering=0) as handle:
            handle.seek(start)
            handle.write(payload)
        # The marker is written only after the complete chunk reaches disk.
        with write_lock:
            marker.touch()
        return index, len(payload)

    pending = [index for index in range(chunks) if not (done_dir / f"{index}.done").exists()]
    print(
        f"{archive.name}: {completed_bytes / 1024**3:.2f}/{archive.size / 1024**3:.2f} GiB "
        f"already complete; {len(pending)} chunks pending",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in pending]
        for number, future in enumerate(as_completed(futures), 1):
            _, count = future.result()
            completed_bytes += count
            if number % max(1, workers) == 0 or number == len(futures):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (completed_bytes / elapsed) / 1024**2
                print(
                    f"{archive.name}: {completed_bytes / archive.size:6.1%} "
                    f"({rate:.2f} MiB/s current-run average)",
                    flush=True,
                )

    if destination.stat().st_size != archive.size:
        raise IOError(f"{destination} has the wrong size")
    if verify:
        actual = md5sum(destination)
        if actual != archive.md5:
            raise IOError(f"MD5 mismatch for {destination}: {actual} != {archive.md5}")
    shutil.rmtree(done_dir, ignore_errors=True)
    print(f"complete {destination}", flush=True)
    return destination


def main() -> None:
    default_root = Path(os.environ.get("DENSITY_EQGNN_DATA_ROOT", r"D:\density-eqgnn-data"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_root / "qm9" / "archives")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--archive",
        action="append",
        choices=[archive.name for archive in ARCHIVES],
        help="Download only the named shard; may be repeated",
    )
    args = parser.parse_args()
    selected = (
        [archive for archive in ARCHIVES if archive.name in args.archive]
        if args.archive
        else ARCHIVES
    )
    for archive in selected:
        download_archive(
            archive,
            args.output,
            workers=args.workers,
            chunk_size=args.chunk_mib * 1024 * 1024,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
