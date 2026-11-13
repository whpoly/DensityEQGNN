"""Local sequential download/evaluation pipeline; survives closing the tool turn."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path('runs/public_benchmark_2026-10-30')
WORKSPACE=Path(__file__).resolve().parents[1]


def save(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=('nmc','ec'),default=['nmc','ec'])
    p.add_argument('--wait-nmc-download-pid',type=int)
    args=p.parse_args();os.chdir(WORKSPACE);ROOT.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'pipeline.lock'
    if lock.exists():
        old=json.loads(lock.read_text())
        if psutil.pid_exists(old['pid']):
            process=psutil.Process(old['pid'])
            if abs(process.create_time()-old['created'])<.01:raise RuntimeError('Pipeline already running')
        lock.unlink()
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    status=dict(pid=os.getpid(),datasets=args.datasets,started_utc=datetime.now(timezone.utc).isoformat())
    def update(stage,**more):
        status.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**more)
        save(ROOT/'pipeline_status.json',status);print(json.dumps(status),flush=True)
    # Reject source edits while waiting for large downloads as well as during inference.
    from scripts.benchmark_public_density import SOURCES
    files=SOURCES+['scripts/download_public_density.py','scripts/summarize_public_density.py',
                   'scripts/run_public_density.py','reports/public_benchmark_2026-10-30/protocol.md']
    verification=json.loads((ROOT/'reader_adapter_verification.json').read_text())
    assert verification['passed']
    assert all(sha(s)==h for s,h in verification['source_hashes'].items())
    hashes={s:sha(s) for s in files}
    manifest=ROOT/'pipeline_sources.json'
    if manifest.exists():assert json.loads(manifest.read_text())==hashes,'Use a new run for changed sources'
    else:save(manifest,hashes)
    def check():assert all(sha(s)==h for s,h in hashes.items()),'Pipeline sources changed'
    def run(label,module,*options):
        check();update(label)
        path=ROOT/(label+'.log')
        with path.open('a',encoding='utf-8') as out:
            out.write('\nRUN '+datetime.now(timezone.utc).isoformat()+'\n');out.flush()
            process=subprocess.Popen([sys.executable,'-u','-m',module,*options],stdout=out,stderr=subprocess.STDOUT,cwd=WORKSPACE)
            update(label,child_pid=process.pid)
            with (ROOT/'gpu_telemetry.log').open('a',encoding='utf-8') as telemetry:
                while True:
                    telemetry.write(datetime.now(timezone.utc).isoformat()+' '+label+'\n');telemetry.flush()
                    subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader'],stdout=telemetry,stderr=telemetry,timeout=10)
                    subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],stdout=telemetry,stderr=telemetry,timeout=10)
                    try:code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:pass
            if code:raise RuntimeError(f'{label} exited {code}; see {path}')
        check()
    try:
        if args.wait_nmc_download_pid:
            try:
                process=psutil.Process(args.wait_nmc_download_pid)
                command=process.cmdline()
                if 'scripts.download_public_density' in command and 'nmc' in command:
                    update('waiting_existing_nmc_download',child_pid=process.pid)
                    while process.is_running():
                        try:process.wait(timeout=30)
                        except psutil.TimeoutExpired:check()
            except psutil.NoSuchProcess:pass
        for dataset in args.datasets:
            run(dataset+'_download','scripts.download_public_density','--dataset',dataset,'--workers','8')
            run(dataset+'_prepare','scripts.benchmark_public_density','--dataset',dataset,'--stage','prepare')
            run(dataset+'_accuracy','scripts.benchmark_public_density','--dataset',dataset,'--stage','accuracy')
            for variant in ('reference','fused_same_chunk','fused'):
                run(dataset+'_timing_'+variant,'scripts.benchmark_public_density','--dataset',dataset,'--stage','timing','--variant',variant)
            run(dataset+'_summary','scripts.summarize_public_density','--dataset',dataset)
            result=json.loads((Path('reports/public_benchmark_2026-10-30')/f'{dataset}_summary.json').read_text())
            if not result['all_pass']:raise RuntimeError(f'{dataset}: reported failures; inspect before continuing')
        update('complete',child_pid=None)
    except BaseException as error:
        update('failed',error=str(error));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
