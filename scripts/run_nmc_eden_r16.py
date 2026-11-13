"""One-shot serial execution of the two predefined Eden-inspired experiments."""
from datetime import datetime,timezone
import json
import subprocess
import sys
from scripts.nmc_eden_r16 import ROOT,ARMS
from scripts.nmc_edengnn import write_json


def status(phase,**kwargs):
    data={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**kwargs}
    write_json(ROOT/'pipeline_status.json',data)
    print(json.dumps(data),flush=True)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:
        status('training',arm=arm)
        command=[sys.executable,'-u','-m','scripts.train_nmc_eden_r16','--arm',arm]
        if (ROOT/arm/'settings.json').exists():
            command.append('--resume')
        with (ROOT/f'{arm}.stdout.log').open('a') as out,(ROOT/f'{arm}.stderr.log').open('a') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err)
            (ROOT/'training.pid').write_text(str(child.pid))
            if child.wait():
                raise RuntimeError(f'{arm} training failed')
    status('evaluation_and_timing')
    with (ROOT/'final.stdout.log').open('a') as out,(ROOT/'final.stderr.log').open('a') as err:
        subprocess.run([sys.executable,'-u','-m','scripts.finalize_nmc_eden_r16'],stdout=out,stderr=err,check=True)
    status('complete',report='reports/nmc_eden_r16_2026-10-29/report.md')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
