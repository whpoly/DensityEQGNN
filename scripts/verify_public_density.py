"""Validate new archive reader and adapter against upstream on training frames."""
import gc
from pathlib import Path
import torch

from scripts.benchmark_public_density import ROOT,SOURCES,decode,model_for,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.nmc_probe_holdout import PeriodicQueryGraph,atom_batch,probe_batch,sha256


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for dataset,path in [('nmc',Path('data/nmc_cache/0.CHGCAR.lz4')),
                         ('ec',Path('data/public_density/ec/preflight/01000.CHGCAR.lz4'))]:
        sample,target=decode(path.read_bytes(),path.name)
        prior=load_sample(path,'cuda')
        for key in ('positions','numbers','cell','probes'):
            torch.testing.assert_close(sample[key],prior[key],rtol=0,atol=0)
        torch.testing.assert_close(target,prior['target'],rtol=0,atol=0)
        del prior
        model,_=model_for(dataset)
        reference=flat_density(model,FlatProbe(model),sample,chunk=4096)
        # Uniformly spaced queries include both ends of the original full grid.
        indices=torch.linspace(0,len(target)-1,2048,device='cuda').long()
        mini=dict(sample,probes=sample['probes'][indices])
        mini['batch']=atom_batch(mini)
        mini['graph']=PeriodicQueryGraph(mini['positions'],mini['probes'],mini['cell'],4.)
        upstream=model(probe_batch(mini,torch.arange(len(indices),device='cuda'))).reshape(-1)
        actual=reference[indices]
        torch.testing.assert_close(actual,upstream,rtol=5e-5,atol=5e-5)
        relative=float((actual.double()-upstream.double()).abs().sum()/upstream.double().abs().sum())
        assert relative<1e-6
        row=dict(dataset=dataset,training_member=path.name,reader_exact=True,
                 upstream_queries=len(indices),relative_l1=relative,passed=True)
        rows.append(row);print(row,flush=True)
        del model,mini,sample,target,upstream,actual,reference;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'reader_adapter_verification.json',dict(passed=True,records=rows,
        source_hashes={s:sha256(s) for s in SOURCES},verification_sha256=sha256(__file__)))


if __name__=='__main__':main()
