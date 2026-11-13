"""Check real full fields, new coordinates and shape-cache eviction."""
from pathlib import Path
import json
import torch
from scripts.predict_adaptive_r16_replay import ReplayAdaptiveR16
from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_edengnn import write_json
from scripts.benchmark_nmc_edengnn import timed


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    predictor=ReplayAdaptiveR16(max_graphs=1)
    samples=[load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda') for sid in (0,1)]
    result={'records':[]}
    previous=None
    for index,sid in enumerate((0,1,0)):
        s=samples[sid]
        args=(s['numbers'],s['positions'],s['cell'],s['metadata']['grid_shape'])
        timing,actual=timed(lambda:predictor(*args),1,warmup=0)
        expected=FastAdaptiveR16.__call__(predictor,*args)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=5e-5)
        assert len(predictor.graphs.cache)==1
        assert len(predictor.graphs.captures)==index+1
        if previous is not None:
            torch.testing.assert_close(previous[0],previous[1],rtol=0,atol=0)
        previous=(actual,actual.clone())
        result['records'].append({'sample_id':sid,'first_call_seconds':timing['median_seconds'],
                                  'max_abs_difference':float((actual-expected).abs().max()),
                                  'captures':len(predictor.graphs.captures)})
    s=samples[0]
    moved=s['positions'].clone()
    moved[0]+=torch.tensor([.001,-.002,.0015],device='cuda')
    args=(s['numbers'],moved,s['cell'],s['metadata']['grid_shape'])
    actual=predictor(*args)
    expected=FastAdaptiveR16.__call__(predictor,*args)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=5e-5)
    result['changed_coordinates_max_abs_difference']=float((actual-expected).abs().max())
    result['captures_after_changed_coordinates']=len(predictor.graphs.captures)
    result['old_outputs_preserved']=True
    predictor.clear_graphs()
    assert not predictor.graphs.cache
    result['clear_cache_passed']=True
    write_json('runs/nmc_inference_graph_2026-10-29/api_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
