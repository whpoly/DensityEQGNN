"""Train-partition-only small-norm LS teachers for R16 pretraining."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import torch
from density_eqgnn.compact_basis import basis_checkpoint_metadata
from density_eqgnn.oracle_basis import FullGridDesign,OracleBasis
from scripts.fit_nmc_holdout_oracle import BASIS,degree_to_radial,independent_metrics
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import DEFAULT_ROOT,save_split,sha256
from scripts.nmc_edengnn import write_json

ROOT=Path('runs/nmc_coefficient_pretrain_2026-10-29')


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    ROOT.mkdir(parents=True,exist_ok=True)
    for sid in range(8):
        folder=ROOT/'teachers'/f'sample_{sid:04d}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            completed=json.loads((folder/'complete.json').read_text())
            assert completed['checkpoint_sha256']==sha256(folder/'teacher.pt')
            print(f'SKIP teacher={sid}',flush=True)
            continue
        start=time.perf_counter()
        write_json(ROOT/'status.json',{'phase':'teacher_fit','sample_id':sid,'updated_utc':datetime.now(timezone.utc).isoformat()})
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        split,metadata=save_split(DEFAULT_ROOT/'spatial_splits',len(sample['target']),sid,sample['metadata']['sha256'])
        old=json.loads(Path(f'runs/nmc_bottleneck_2026-10-27/oracle/sample_{sid:04d}/settings.json').read_text())
        assert metadata==old['split'] and asdict(BASIS)==old['basis']
        train=torch.as_tensor(split['train'],device='cuda')
        target=sample['target'][train]
        normalizer=target.abs().mean()
        design=FullGridDesign(OracleBasis([(sample['positions'],BASIS.radial_spec)],sample['cell']),sample['probes'][train])
        scales=design.normalize_columns().double()
        print(f'DESIGN sample={sid} seconds={time.perf_counter()-start:.1f}',flush=True)
        factor=design.gram_cholesky(1e-9)
        rhs=design.precise_transpose(target/normalizer)
        solution=torch.cholesky_solve(rhs[:,None],factor).squeeze(1)
        coefficients=solution/scales*normalizer
        rms=float(coefficients.square().mean().sqrt())
        train_nmae=float((design.precise_matvec(solution)-target/normalizer).abs().mean())
        history=json.loads(Path(f'runs/nmc_bottleneck_2026-10-27/oracle/sample_{sid:04d}/history.json').read_text())
        assert abs(train_nmae-history[0]['train_nmae'])<2e-5
        assert abs(rms-history[0]['coefficient_rms'])/rms<2e-4
        state={'coefficients':degree_to_radial(coefficients,len(sample['numbers'])).cpu(),
               'degree_coefficients':coefficients.cpu(),'layout':'batch,atom,radial,m',
               **basis_checkpoint_metadata(BASIS),'split':metadata,'source':sample['metadata'],
               'ridge':1e-9,'training_normalizer':float(normalizer),'coefficient_rms':rms,
               'train_nmae_design':train_nmae,'fit_points':len(train),
               'validation_points_used_for_fit':0,'test_points_used_for_fit':0,
               'frozen_before_validation_utc':datetime.now(timezone.utc).isoformat(),
               'source_sha256':{str(p):sha256(p) for p in [Path(__file__),Path('src/density_eqgnn/oracle_basis.py')]}}
        torch.save(state,folder/'teacher.pt')
        digest=sha256(folder/'teacher.pt')
        del design,factor,rhs,solution,scales
        gc.collect()
        torch.cuda.empty_cache()
        metrics=independent_metrics(sample,coefficients,split,('selection','validation'),2048)
        assert abs(metrics['selection_nmae']-history[0]['selection_nmae'])<2e-5
        record={'sample_id':sid,**metrics,'coefficient_rms':rms,'train_nmae_design':train_nmae,
                'checkpoint_sha256':digest,'elapsed_seconds':time.perf_counter()-start,
                'validation_points_used_for_fit':0,'test_points_used_for_fit':0}
        write_json(folder/'complete.json',record)
        print('TEACHER_COMPLETE '+json.dumps(record),flush=True)
        del sample,state,coefficients,train,target,normalizer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(ROOT/'teachers_complete.json',{'complete':True,'sample_ids':list(range(8))})


if __name__=='__main__':
    main()
