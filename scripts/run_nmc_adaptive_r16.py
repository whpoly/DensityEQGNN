"""Durable serial pipeline: paired training, then frozen evaluation and timings."""
from datetime import datetime,timezone
import subprocess
import sys
from scripts.nmc_adaptive_r16 import ROOT,ARMS
from scripts.nmc_edengnn import write_json


def status(phase,**kwargs):
    write_json(ROOT/'pipeline_status.json',{'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**kwargs})


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:
        status('training',arm=arm)
        command=[sys.executable,'-u','-m','scripts.train_nmc_adaptive_r16','--arm',arm]
        if (ROOT/arm/'settings.json').exists():
            command.append('--resume')
        with (ROOT/f'{arm}.stdout.log').open('a') as out,(ROOT/f'{arm}.stderr.log').open('a') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err)
            (ROOT/'training.pid').write_text(str(child.pid))
            if child.wait():
                raise RuntimeError(f'{arm} training failed')
    status('evaluation_and_timing')
    with (ROOT/'final.stdout.log').open('a') as out,(ROOT/'final.stderr.log').open('a') as err:
        subprocess.run([sys.executable,'-u','-m','scripts.finalize_nmc_adaptive_r16'],stdout=out,stderr=err,check=True)
    status('complete',report='reports/nmc_adaptive_r16_2026-10-29/report.md')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
