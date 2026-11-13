"""One-shot serial teacher -> two predeclared arms -> frozen final evaluation."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
from scripts.prepare_nmc_coefficient_teachers import ROOT
from scripts.nmc_edengnn import write_json


def status(phase,**kwargs):
    data={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**kwargs}
    write_json(ROOT/'pipeline_status.json',data)
    print(json.dumps(data),flush=True)


def main():
    if not (ROOT/'teachers_complete.json').exists():
        process=psutil.Process(int((ROOT/'teachers.pid').read_text()))
        assert 'scripts.prepare_nmc_coefficient_teachers' in process.cmdline()
        status('waiting_for_teachers',pid=process.pid)
        while process.is_running():
            time.sleep(10)
        if not (ROOT/'teachers_complete.json').exists():
            raise RuntimeError('Teacher fitting exited without completion')
    for arm in ['coefficient_mse','density_metric']:
        status('training',arm=arm)
        arguments=[sys.executable,'-u','-m','scripts.train_nmc_coefficient_pretrain','--arm',arm]
        if (ROOT/arm/'settings.json').exists():
            arguments.append('--resume')
        with (ROOT/f'{arm}.stdout.log').open('a') as out,(ROOT/f'{arm}.stderr.log').open('a') as err:
            child=subprocess.Popen(arguments,stdout=out,stderr=err)
            (ROOT/'training.pid').write_text(str(child.pid))
            code=child.wait()
            if code:
                raise RuntimeError(f'{arm} exited with code {code}')
    status('final_evaluation')
    with (ROOT/'final.stdout.log').open('a') as out,(ROOT/'final.stderr.log').open('a') as err:
        subprocess.run([sys.executable,'-u','-m','scripts.finalize_nmc_coefficient_pretrain'],stdout=out,stderr=err,check=True)
    status('complete',report='reports/nmc_coefficient_pretrain_2026-10-29/report.md')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
