"""Smoke test with a frozen checkpoint, changed inputs and synchronized timing."""
from pathlib import Path
import json
import torch
from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_edengnn import atom_data,write_json
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_inference_fast import decode_grouped
from scripts.r16_inference_graph import GraphCoefficients,decode_uniform,GraphUniformDecoder
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.benchmark_nmc_edengnn import timed

ROOT=Path('runs/nmc_inference_graph_2026-10-29')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    predictor=FastAdaptiveR16()
    model=predictor.model
    s=load_sample(Path('data/nmc_cache/0.CHGCAR.lz4'),'cuda')
    data=atom_data(s)
    graph=GraphCoefficients(model,max_graphs=1)
    result={}
    result['eager_network'],reference=timed(lambda:model.coefficients(data),3)
    result['first_graph_call'],first=timed(lambda:graph(data),1,warmup=0)
    result['graph_network'],replay=timed(lambda:graph(data),5)
    torch.testing.assert_close(first,reference,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(replay,reference,rtol=2e-5,atol=2e-6)
    result['coefficient_max_abs']=float((replay-reference).abs().max())
    retained=replay.clone()
    # Same shape, but changed species, positions, topology order and edge shifts.
    changed=data.clone()
    changed.pos=changed.pos+torch.randn_like(changed.pos)*.005
    changed.z=changed.z.roll(1)
    changed.edge_index=changed.edge_index.flip(1)
    changed.nbr_shift=changed.nbr_shift.flip(0)+.001
    other=graph(changed)
    eager_other=model.coefficients(changed)
    torch.testing.assert_close(other,eager_other,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(replay,retained,rtol=0,atol=0)
    assert not torch.allclose(other,replay)
    torch.testing.assert_close(graph(data),reference,rtol=2e-5,atol=2e-6)
    result['changed_input_max_abs']=float((other-eager_other).abs().max())
    result['previous_output_preserved']=True
    result['capture_records']=graph.captures
    c=model.coefficients(data)
    def previous():
        e=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
        return decode_grouped(c,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS)
    result['old_geometry_and_decoder'],old=timed(previous,3)
    result['direct_geometry_and_decoder'],direct=timed(lambda:decode_uniform(
        c,s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS),3)
    torch.testing.assert_close(direct,old,rtol=3e-5,atol=5e-5)
    result['decoder_max_abs']=float((direct-old).abs().max())
    decoder=GraphUniformDecoder(BASIS,max_graphs=1)
    args=(c,s['positions'],s['cell'],s['metadata']['grid_shape'])
    result['first_decoder_graph_call'],first=timed(lambda:decoder(*args),1,warmup=0)
    result['decoder_graph'],replayed=timed(lambda:decoder(*args),5)
    torch.testing.assert_close(replayed,old,rtol=3e-5,atol=5e-5)
    result['decoder_graph_max_abs']=float((replayed-old).abs().max())
    # Replaying the same graph must consume new learned coefficients and geometry.
    moved=s['positions']+.001*torch.randn_like(s['positions'])
    altered=c*.99
    expected=decode_uniform(altered,moved,s['cell'],s['metadata']['grid_shape'],BASIS)
    actual=decoder(altered,moved,s['cell'],s['metadata']['grid_shape'])
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=5e-5)
    assert not torch.allclose(actual,replayed)
    torch.testing.assert_close(replayed,first,rtol=3e-5,atol=5e-5)
    result['changed_decoder_inputs_max_abs']=float((actual-expected).abs().max())
    result['decoder_capture_records']=decoder.captures
    write_json(ROOT/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
