"""CUDA-event stage diagnostic; separate from uninstrumented wall-time tests."""
import json
from pathlib import Path
import time

import torch

from scripts.benchmark_deepdft_reuse import CHECKPOINT,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe,ProbeEdges
from scripts.deepdft_cubic_far import periodic_expand,expanded_far_density
from scripts.deepdft_near_far import far_fields,weighted_geometry
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,atom_batch


@torch.no_grad()
def profile(model,core,sample,chunk=4096):
    events = []
    def part(name,fn):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        value = fn()
        b.record()
        events.append((name,a,b))
        return value
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    batch = part('atom_graph',lambda:atom_batch(sample))
    rep = part('atom_network',lambda:model.atom_model(batch))
    states = part('atom_projection',lambda:core.atoms(rep))
    scalars,vectors,projections = states
    width = scalars.shape[-1]
    shape = tuple(sample['metadata']['grid_shape'])
    coarse = part('far_graph_and_messages',lambda:far_fields(core,sample,states,3.,1.,2,chunk))
    fields = part('far_interpolation',lambda:[periodic_expand(coarse[k].reshape(
        *(n//2 for n in shape),4*width),2,'cubic').reshape(-1,4*width) for k in range(len(scalars))])
    del coarse
    graph = part('near_graph',lambda:ProbeEdges.build(sample,cutoff=3.))
    out = scalars.new_empty(graph.size)
    for start in range(0,graph.size,chunk):
        stop = min(start+chunk,graph.size)
        size = stop-start
        source,receiver,diff = part('near_geometry',lambda:graph.chunk(sample,start,stop))
        geometry = part('near_radial',lambda:weighted_geometry(core,diff,3.,1.))
        s,v = scalars.new_zeros(size,width),vectors.new_zeros(size,3,width)
        for k,(layer,update) in enumerate(zip(core.probe.message_layers,core.probe.scalar_vector_update)):
            a,b = part('near_messages',lambda:core.aggregate(layer,scalars[k],vectors[k],projections[k],source,receiver,geometry,size))
            def update_query():
                shared = fields[k][start:stop]
                ms,mv = a+shared[:,:width],b+shared[:,width:].reshape(size,3,width)
                gs,gv = layer.update_gate_mlp(ms).split(width,1)
                return update(gs*s+(1-gs)*ms,gv[:,None]*v+(1-gv[:,None])*mv)
            s,v = part('query_updates',update_query)
        out[start:stop] = part('readout',lambda:core.probe.readout_function(s).squeeze(1))
    torch.cuda.synchronize()
    wall = time.perf_counter()-start_time
    stats = {}
    for name,a,b in events:
        stats[name] = stats.get(name,0.)+a.elapsed_time(b)/1000
    return dict(instrumented_wall_seconds=wall,stage_cuda_seconds=stats)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model,_ = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    core = FlatProbe(model).eval()
    result = []
    for sid in (0,3):
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        expanded_far_density(model,core,sample,radius=3.)
        row = dict(id=sid,**profile(model,core,sample))
        result.append(row)
        save(Path('runs/deepdft_cubic_fast_2026-10-30/profile.json'),result)
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
