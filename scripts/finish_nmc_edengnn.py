"""One-shot serial continuation: training exit -> full benchmark -> report."""
from __future__ import annotations

from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path('runs/edengnn_nmc_2026-10-28')


def status(phase,**details):
    payload={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**details}
    temporary=ROOT/'pipeline_status.tmp'
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    temporary.replace(ROOT/'pipeline_status.json')
    print(json.dumps(payload),flush=True)


def main():
    pid=int((ROOT/'training.pid').read_text())
    try:
        process=psutil.Process(pid)
        assert any('scripts.train_nmc_edengnn'==x or 'scripts.edengnn_graph_training'==x for x in process.cmdline())
    except psutil.NoSuchProcess:
        process=None
    status('waiting_for_training',training_pid=pid)
    while process is not None and process.is_running():
        time.sleep(10)
    state=json.loads((ROOT/'eden_eight/status.json').read_text())
    if state['phase']!='complete':
        raise RuntimeError(f'Training exited without completion: {state}')
    status('benchmark')
    with (ROOT/'benchmark.stdout.log').open('w') as out,(ROOT/'benchmark.stderr.log').open('w') as err:
        subprocess.run([sys.executable,'-u','-m','scripts.benchmark_nmc_edengnn','--repeat','5'],stdout=out,stderr=err,check=True)
    status('report')
    subprocess.run([sys.executable,'-u','-m','scripts.summarize_nmc_edengnn'],check=True)
    status('complete',report='reports/nmc_edengnn_2026-10-28/report.md')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
