"""Verified, resumable downloads of official density archives (no QM9)."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

DISCOVERY = Path('runs/public_benchmark_2026-10-30/discovery')


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def fetch_range(url,start,stop,total):
    for attempt in range(5):
        try:
            with requests.get(url,headers={'Range':f'bytes={start}-{stop}'},
                              timeout=(15,60),stream=True) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{stop}/{total}':
                    raise ValueError('Server did not return the exact requested range')
                payload=r.raw.read(stop-start+2)
                if len(payload)!=stop-start+1:raise ValueError('Truncated range')
                return payload
        except (requests.RequestException,ValueError,OSError):
            if attempt==4:raise
            time.sleep(min(2**attempt,8))


def download(record,directory,workers=8):
    name=record.get('name',record.get('key'))
    assert Path(name).name==name
    size=record['size']
    expected=record.get('computed_md5',record.get('checksum','').removeprefix('md5:'))
    assert len(expected)==32
    url=record.get('download_url',record.get('links',{}).get('self'))
    path=directory/name
    receipt=directory/(name+'.verified.json')
    if receipt.exists():
        assert path.stat().st_size==size and digest(path,'md5')==expected
        print('VERIFIED_REUSE '+name,flush=True)
        return
    part=directory/(name+'.part')
    markers=directory/(name+'.ranges')
    markers.mkdir(exist_ok=True)
    if path.exists():
        assert path.stat().st_size==size and digest(path,'md5')==expected
    else:
        if not part.exists():
            if shutil.disk_usage(directory).free<size+20*1024**3:
                raise OSError('Insufficient free disk space including 20 GiB reserve')
            with part.open('xb') as f:f.truncate(size)
        assert part.stat().st_size==size
        chunk=8*1024**2
        pending=[]
        ready=0
        with part.open('rb') as f:
            for start in range(0,size,chunk):
                stop=min(start+chunk,size)-1
                marker=markers/f'{start}.sha256'
                if marker.exists():
                    f.seek(start)
                    if hashlib.sha256(f.read(stop-start+1)).hexdigest()==marker.read_text():
                        ready+=stop-start+1
                        continue
                pending.append((start,stop,marker))
        begin=time.perf_counter();initial=ready;last=begin
        print(f'START {name} {ready/2**30:.3f}/{size/2**30:.3f} GiB',flush=True)
        def one(item):
            start,stop,marker=item
            data=fetch_range(url,start,stop,size)
            with part.open('r+b',buffering=0) as f:f.seek(start);f.write(data)
            marker.write_text(hashlib.sha256(data).hexdigest())
            return len(data)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(one,item) for item in pending]):
                ready+=f.result();now=time.perf_counter()
                if now-last>=15 or ready==size:
                    print(f'PROGRESS {name} {ready/2**30:.3f}/{size/2**30:.3f} GiB '
                          f'{(ready-initial)/2**20/max(now-begin,.001):.2f} MiB/s',flush=True)
                    last=now
        actual=digest(part,'md5')
        if actual!=expected:raise ValueError(f'MD5 mismatch for {name}: {actual}')
        part.rename(path)
    receipt.write_text(json.dumps(dict(name=name,bytes=size,md5=expected,
        sha256=digest(path),source_url=url,verified=True),indent=2),encoding='utf-8')
    print('VERIFIED '+name,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=('nmc','ec','eden'),required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    directory=Path('data/public_density')/args.dataset
    directory.mkdir(parents=True,exist_ok=True)
    data=json.loads((DISCOVERY/f'{args.dataset}.json').read_text())
    if args.dataset=='eden':
        records=[r for r in data['files'] if r['key'] in ('data.zip','model_weight.zip')]
    else:
        records=sorted([r for r in data['files'] if r['name'].endswith('.tar')],key=lambda r:r['name'])
        if args.dataset=='ec':
            # Official split index 0 is trajectory frame 1000; the first 1000
            # non-equilibrated frames are excluded by the authors' file list.
            names=(DISCOVERY/'ec_filelist.txt').read_text().splitlines()
            by_name={r['name']:r for r in records}
            records=[by_name[n] for n in names if n.strip()]
        shutil.copyfile(DISCOVERY/f'{args.dataset}_split.json',directory/'split.json')
    for record in records:download(record,directory,args.workers)
    print('COMPLETE '+args.dataset,flush=True)


if __name__=='__main__':main()
