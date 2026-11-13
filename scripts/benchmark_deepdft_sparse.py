"""Predefined 2x-per-axis probe sharing diagnostic; no retraining or tuning.

This is approximate spatial interpolation, separate from the exact reuse study.
Both raw density and SAD-residual interpolation are reported, without clipping.
"""
import json
from pathlib import Path
import numpy as np
import torch

from scripts.benchmark_deepdft_reuse import ROOT,CHECKPOINT,measure,errors,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.r16_inference_fast import TorchSAD
from scripts.evaluate_vasp_sad_nmc import read_form_factors


def periodic_double(values):
    """Separable periodic linear interpolation, preserving all even-index nodes."""
    for axis in range(3):
        shape=list(values.shape);shape[axis]*=2
        out=values.new_empty(shape)
        even=[slice(None)]*3;odd=list(even)
        even[axis]=slice(0,None,2);odd[axis]=slice(1,None,2)
        out[tuple(even)]=values
        out[tuple(odd)]=.5*(values+torch.roll(values,-1,axis))
        values=out
    return values


def sparse(model,core,sample,sad=None):
    shape=tuple(sample['metadata']['grid_shape'])
    if any(n%2 for n in shape):raise ValueError('Even grid sizes required')
    coarse_shape=tuple(n//2 for n in shape)
    coarse=dict(sample,probes=sample['probes'].reshape(*shape,3)[::2,::2,::2].reshape(-1,3),
                metadata=dict(sample['metadata'],grid_shape=coarse_shape))
    prediction=flat_density(model,core,coarse,chunk=4096).reshape(coarse_shape)
    if sad is None:return periodic_double(prediction).flatten()
    baseline=sad(sample).reshape(shape)
    return (periodic_double(prediction-baseline[::2,::2,::2])+baseline).flatten()


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,_=official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model']);model.eval()
    core=FlatProbe(model).eval();sad=TorchSAD(read_form_factors(Path('potpaw54.zip')))
    result={'checkpoint_sha256':sha256(CHECKPOINT),'stride_per_axis':2,'probe_fraction':.125,
            'interpolation':'periodic trilinear; no training, clipping, renormalization, or test-based tuning',
            'records':[]}
    for sid in range(8):
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        reference=flat_density(model,core,sample,chunk=4096)
        row={'id':sid,'variants':{}}
        functions={'full':lambda:flat_density(model,core,sample,chunk=4096),
                   'sparse_raw':lambda:sparse(model,core,sample),
                   'sparse_sad_residual':lambda:sparse(model,core,sample,sad)}
        for name,fn in functions.items():
            out=fn()
            row['variants'][name]={'seconds':[],**errors(out,reference,sample,sid)}
            if name!='full':
                shape=tuple(sample['metadata']['grid_shape'])
                torch.testing.assert_close(out.reshape(shape)[::2,::2,::2],
                    reference.reshape(shape)[::2,::2,::2],rtol=3e-5,atol=5e-5)
            del out
        names=list(functions)
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                stats,out=measure(functions[name],repeat=1,warmup=0)
                row['variants'][name]['seconds']+=stats['seconds'];del out
        for value in row['variants'].values():value['median_seconds']=float(np.median(value['seconds']))
        result['records'].append(row);save(ROOT/'sparse_diagnostic.json',result)
        print('SPARSE '+json.dumps(row),flush=True)


if __name__=='__main__':main()
