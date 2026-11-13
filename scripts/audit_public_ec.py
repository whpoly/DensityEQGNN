"""Audit official EC checkpoint on training geometry; no test-driven tuning."""
import gc
import json
from pathlib import Path
import time

import torch

from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256

ROOT=Path('runs/public_benchmark_2026-10-30')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    destination=ROOT/'ec_preflight.json'
    assert not destination.exists(),'Preserve existing preflight'
    repository=DEFAULT_ROOT/'DeepDFT'
    path=repository/'pretrained_models/ethylenecarbonate_painn/best_model.pth'
    model,provenance=official_model(repository,'cuda')
    saved=torch.load(path,map_location='cuda',weights_only=False)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    loaded=load_sample(Path('data/public_density/ec/preflight/01000.CHGCAR.lz4'),'cuda')
    sample={k:loaded[k] for k in ('positions','numbers','cell','probes','metadata')}
    reference=flat_density(model,FlatProbe(model),sample,chunk=4096)
    records=[]
    for name,chunk in [('reference',4096),('fused',4096),('fused',8192)]:
        core=FlatProbe(model) if name=='reference' else RadialTableProbe(model,fused=True)
        actual=flat_density(model,core,sample,chunk=chunk)
        error=(actual.double()-reference.double()).abs()
        relative=float(error.sum()/reference.double().abs().sum())
        torch.testing.assert_close(actual,reference,rtol=5e-5,atol=5e-5)
        assert relative<1e-6
        nmae=float((actual.double()-loaded['target']).abs().sum()/loaded['target'].abs().sum())
        del actual,error
        for _ in range(2):
            value=flat_density(model,core,sample,chunk=chunk);del value
        seconds=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            value=flat_density(model,core,sample,chunk=chunk)
            torch.cuda.synchronize();seconds.append(time.perf_counter()-start);del value
        row=dict(name=name,chunk=chunk,relative_l1=relative,nmae=nmae,seconds=seconds)
        records.append(row);print(json.dumps(row),flush=True)
        del core;gc.collect();torch.cuda.empty_cache()
    destination.write_text(json.dumps(dict(passed=True,scope='EC official checkpoint; training split index 0 / trajectory frame 1000 only; member fetched over HTTPS; full archive MD5 pending',
        checkpoint=str(path),checkpoint_sha256=sha256(path),provenance=provenance,records=records,
        geometry=loaded['metadata'],device=torch.cuda.get_device_name(),torch=torch.__version__,
        source_hashes={str(p):sha256(p) for p in [Path(__file__),Path('scripts/deepdft_radial_table.py'),
            Path('scripts/radial_message_kernel.cu')]}),indent=2),encoding='utf-8')


if __name__=='__main__':main()
