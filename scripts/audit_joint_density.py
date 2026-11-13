"""Training-geometry-only numerical audit before validation/test selection."""
import gc
from pathlib import Path
import time

import torch

from scripts.benchmark_deepdft_reuse import save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.joint_density_predictor import JointPredictor
from scripts.deepdft_cubic_far import expanded_far_density
from scripts.deepdft_factored_messages import FactoredProbe
from scripts.deepdft_paw import density_and_paw
from scripts.deepdft_inference_fast import flat_density,FlatProbe
from scripts.nmc_probe_holdout import sha256

ROOT = Path('runs/joint_density_benchmark_2026-10-30')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if (ROOT/'numerical_audit.json').exists():
        raise ValueError('Preserve existing audit')
    loaded = load_sample(Path('data/nmc_cache/0.CHGCAR.lz4'),'cuda')
    sample = {k:loaded[k] for k in ('positions','numbers','cell','probes','metadata')}
    del loaded
    records,references = [],{}
    for name in ('deep_reference','deep_factored','deep_table','deep_fused','fused_cubic_r3','fused_cubic_r3.25','fused_cubic_r3.5'):
        predictor = JointPredictor(name)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = predictor.predict(sample)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        if name=='deep_reference':
            independent = density_and_paw(predictor.model,predictor.head,sample)
            for k in output:
                torch.testing.assert_close(output[k],independent[k],rtol=3e-5,atol=5e-5)
            del independent
            references = {k:v.clone() for k,v in output.items()}
        radius = predictor.config.get('radius')
        expected = references['density'] if radius is None else expanded_far_density(predictor.model,
            FactoredProbe(predictor.model),sample,radius=radius,chunk=8192)
        rho = output['density']
        relative = float((rho.double()-expected.double()).abs().sum()/expected.double().abs().sum())
        torch.testing.assert_close(rho,expected,rtol=5e-5,atol=5e-5)
        assert relative<1e-6,relative
        torch.testing.assert_close(output['augmentation'],references['augmentation'],rtol=2e-4,atol=2e-5)
        aug_relative = float((output['augmentation'].double()-references['augmentation'].double()).abs().sum()
            /references['augmentation'].double().abs().sum())
        assert aug_relative<1e-6,aug_relative
        record = dict(name=name,max_abs=float((rho-expected).abs().max()),relative_l1=relative,
            unoptimized_smoke_seconds=seconds,occupation_equivalent=True,occupation_relative_l1=aug_relative,
            occupation_max_abs=float((output['augmentation']-references['augmentation']).abs().max()))
        if hasattr(predictor.core,'tables'):
            generator = torch.Generator(device='cuda').manual_seed(42)
            r = torch.cat((torch.rand(32768,device='cuda',generator=generator)*4,
                torch.tensor([0,1e-7,3.99999,4.,4.1],device='cuda')))
            diff = torch.stack((r,r*0,r*0),1)
            _,radial,envelope = FlatProbe(predictor.model).geometry(diff)
            errors = []
            for k,layer in enumerate(predictor.core.probe.message_layers):
                original = layer.filter_layer(radial)*envelope
                actual = predictor.core.lookup(k,r[:,None])
                errors.append(float((actual-original).abs().max()))
                torch.testing.assert_close(actual,original,rtol=5e-5,atol=5e-5)
            record['max_filter_abs'] = max(errors)
            if predictor.core.kernel:
                record['compiler_options'] = predictor.core.kernel.options
            del r,diff,radial,envelope,original,actual
        records.append(record)
        print(record,flush=True)
        del predictor,output,rho,expected
        gc.collect()
        torch.cuda.empty_cache()
    save(ROOT/'numerical_audit.json',dict(passed=True,records=records,
        sources={p:sha256(p) for p in ('scripts/deepdft_radial_table.py','scripts/radial_message_kernel.cu','scripts/joint_density_predictor.py','scripts/audit_joint_density.py')}))


if __name__=='__main__':
    main()
