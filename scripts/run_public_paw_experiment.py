"""Run the locked PAW experiment serially on one GPU; retain per-stage logs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('runs/public_paw_2026-11-02')


def main():
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    stages=[]
    for arm in ('deep_frozen','eden_frozen'):
        stages.append((f'prepare_{arm}',ROOT/f'{arm}_features.json',
            ['-m','scripts.public_paw_experiment','--stage','prepare','--arm',arm]))
        for seed in (42,43):
            stages.append((f'train_{arm}_{seed}',ROOT/f'{arm}_{seed}/selection.json',
                ['-m','scripts.public_paw_experiment','--stage','train','--arm',arm,'--seed',str(seed)]))
    for arm in ('deep_frozen','eden_frozen'):
        stages.append((f'final_{arm}',ROOT/f'{arm}_final.json',
            ['-m','scripts.finalize_public_paw','--stage','final','--arm',arm]))
    stages.append(('verify',ROOT/'joint_verification.json',['-m','scripts.finalize_public_paw','--stage','verify']))
    for mode in ('density','joint','deep_occupation','eden_occupation'):
        stages.append((f'timing_{mode}',ROOT/f'timing_{mode}.json',
            ['-m','scripts.finalize_public_paw','--stage','timing','--mode',mode]))
    for name,marker,args in stages:
        if marker.exists():
            print('SKIP_COMPLETE',name,flush=True);continue
        print('START',name,flush=True)
        start=time.perf_counter()
        with (logs/f'{name}.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u',*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith(('FEATURES','TRAIN','FINAL','VERIFY','TIMING')): print(line,end='',flush=True)
            code=proc.wait()
        if code:
            print('FAILED',name,'see',logs/f'{name}.log',flush=True)
            raise SystemExit(code)
        assert marker.exists(),name
        print('COMPLETE',name,round(time.perf_counter()-start,2),flush=True)
    subprocess.run([sys.executable,'-m','scripts.summarize_public_paw'],check=True)
    print('ALL_SUPPORTED_EXPERIMENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
