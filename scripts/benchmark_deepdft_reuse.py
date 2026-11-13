"""Fixed-checkpoint DeepDFT speed study; no label-based implementation selection."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import (DEFAULT_ROOT, official_model, atom_batch, probe_batch,
                                      PeriodicQueryGraph, spatial_split, sha256)
from scripts.deepdft_inference_fast import FlatProbe, ProbeEdges, ProbeReplay, NeighborReuse, flat_density

ROOT = Path('runs/deepdft_reuse_2026-10-29')
CHECKPOINT = DEFAULT_ROOT/'deepdft_eight/best.pt'


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def measure(fn, repeat=3, warmup=1):
    for _ in range(warmup):
        out=fn()
        del out
    times=[]
    peaks=[]
    for _ in range(repeat):
        torch.cuda.synchronize()
        base=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        out=fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
        peaks.append((torch.cuda.max_memory_allocated()-base)/2**20)
    return {'seconds':times,'median_seconds':float(np.median(times)),
            'peak_increment_mib':max(peaks)},out


def original(model,sample,chunk=2048,geometry=None):
    s=dict(sample)
    s['batch']=atom_batch(s)
    s['graph']=(geometry if geometry is not None else
                PeriodicQueryGraph(s['positions'],s['probes'],s['cell'],4.))
    rep=model.atom_model(s['batch'])
    output=torch.empty(len(s['probes']),device='cuda')
    for ids in torch.arange(len(output),device='cuda').split(chunk):
        output[ids]=model.probe_model(probe_batch(s,ids),*rep)[0]
    return output


def errors(output, reference, sample, sid):
    test=torch.tensor(spatial_split(len(output),sid)['test'],device=output.device)
    target=sample['target']
    return {'max_abs_vs_original':float((output-reference).abs().max()),
            'relative_l1_vs_original':float((output.double()-reference.double()).abs().sum()/reference.double().abs().sum()),
            'test_nmae':float((output[test].double()-target[test]).abs().sum()/target[test].abs().sum()),
            'full_nmae':float((output.double()-target).abs().sum()/target.abs().sum())}


def check(output, reference):
    torch.testing.assert_close(output,reference,rtol=3e-5,atol=5e-5)
    relative=float((output.double()-reference.double()).abs().sum()/reference.double().abs().sum())
    assert relative<1e-6
    return {'max_abs':float((output-reference).abs().max()),'relative_l1':relative}


def profile(model,core,sample,graph,chunk):
    total={}
    def part(name,fn):
        torch.cuda.synchronize();t=time.perf_counter()
        value=fn();torch.cuda.synchronize()
        total[name]=total.get(name,0.)+time.perf_counter()-t
        return value
    batch=part('atom_graph',lambda:atom_batch(sample))
    rep=part('atom_network',lambda:model.atom_model(batch))
    scalars,vectors,projections=part('atom_projections',lambda:core.atoms(rep))
    for start in range(0,graph.size,chunk):
        stop=min(start+chunk,graph.size);size=stop-start
        a,b,d=part('query_geometry',lambda:graph.chunk(sample,start,stop))
        geometry=part('radial_and_direction',lambda:core.geometry(d))
        s=scalars.new_zeros(size,scalars.shape[-1]);v=vectors.new_zeros(size,3,scalars.shape[-1])
        for k,(layer,update) in enumerate(zip(core.probe.message_layers,core.probe.scalar_vector_update)):
            ms,mv=part('edge_messages_and_aggregation',lambda:core.aggregate(layer,scalars[k],vectors[k],projections[k],a,b,geometry,size))
            def query_update():
                gs,gv=layer.update_gate_mlp(ms).split(s.shape[-1],1)
                return update(gs*s+(1.-gs)*ms,gv[:,None]*v+(1.-gv[:,None])*mv)
            s,v=part('query_updates',query_update)
        part('final_readout',lambda:core.probe.readout_function(s))
    return total


@torch.no_grad()
def study(model,core,result,args):
    from scripts.nmc_edengnn import EdenDensity,EdenStencil,atom_data
    from scripts.r16_inference_fast import TorchSAD
    from scripts.evaluate_vasp_sad_nmc import read_form_factors
    from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
    eden=EdenDensity().cuda().eval()
    ep=Path('runs/edengnn_nmc_2026-10-28/eden_eight/best.pt')
    eden.load_state_dict(torch.load(ep,map_location='cuda',weights_only=False)['model'])
    sad=TorchSAD(read_form_factors(Path('potpaw54.zip')))
    hybrid=FastAdaptiveR16()
    result['comparators']={'eden_sha256':sha256(ep),
        'hybrid_sha256':sha256(Path('runs/nmc_adaptive_r16_2026-10-29/adaptive/best.pt'))}
    result['source_hashes']={p:sha256(Path(p)) for p in ['scripts/deepdft_inference_fast.py',__file__]}
    chunk=args.chunk
    for sid in args.samples:
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        print(f'STUDY SAMPLE {sid}',flush=True)
        original_time,reference=measure(lambda:original(model,sample),repeat=1,warmup=0)
        graph_time,graph=measure(lambda:ProbeEdges.build(sample),repeat=3)
        classic_time,classic=measure(lambda:PeriodicQueryGraph(sample['positions'],sample['probes'],sample['cell'],4.),repeat=1)
        def edge_rows(a,b,s):
            x=torch.cat((b[:,None],a[:,None],s.long()),1).cpu().numpy()
            return x[np.lexsort(tuple(x[:,k] for k in range(4,-1,-1)))]
        a,b,s=classic.select(torch.arange(graph.size,device='cuda'))
        if not np.array_equal(edge_rows(a,b,s),edge_rows(graph.source,graph.receiver,graph.shifts)):
            raise AssertionError('Periodic edge multiset mismatch')
        del a,b,s,classic
        cache_time,cache=measure(lambda:NeighborReuse(sample),repeat=1,warmup=0)
        displacement=.005*torch.sin(torch.arange(sample['positions'].numel(),device='cuda').reshape_as(sample['positions']).double()+sid)
        perturbed=dict(sample,positions=sample['positions']+displacement)
        candidate=cache.get(perturbed)
        max_edges=max(int(candidate.offsets[min(i+chunk,graph.size)]-candidate.offsets[i]) for i in range(0,graph.size,chunk))
        torch.cuda.synchronize();t=time.perf_counter()
        replay=ProbeReplay(core,chunk,math_ceil(max_edges,1024),64)
        torch.cuda.synchronize();capture=time.perf_counter()-t
        def eden_fresh():
            local=dict(sample);local['atom_data']=atom_data(sample)
            stencil=EdenStencil(sample,csr=False)
            local['native_data']=local['atom_data'].clone()
            for key,value in stencil.native_fields().items():local['native_data'][key]=value
            local['sad']=sad(sample)
            return eden.native(local)
        functions={
            'official_2048_fresh':lambda:original(model,sample),
            'official_tuned_fresh':lambda:original(model,sample,chunk),
            'flat_fresh':lambda:flat_density(model,core,sample,None,chunk),
            'replay_fresh':lambda:flat_density(model,core,sample,None,chunk,replay),
            'replay_cached_geometry':lambda:flat_density(model,core,sample,graph,chunk,replay),
            'eden_gpu_sad':eden_fresh,
            'hybrid_fast':lambda:hybrid(sample['numbers'],sample['positions'],sample['cell'],sample['metadata']['grid_shape']).flatten().float(),
        }
        row={'id':sid,'metadata':sample['metadata'],'initial_original_call':original_time,
             'local_graph':graph_time,'classic_graph':classic_time,'skin_preparation':cache_time,
             'capture_seconds':capture,'edge_capacity':replay.edge_capacity,
             'exact_edge_multiset':True,'variants':{},'metrics':{}}
        for name,fn in functions.items():
            torch.cuda.empty_cache()
            out=fn()
            if name not in ('eden_gpu_sad','hybrid_fast'):check(out,reference)
            row['metrics'][name]=errors(out,reference,sample,sid)
            del out
        torch.cuda.empty_cache()
        keys=list(functions)
        for r in range(args.repeat):
            for name in keys[r%len(keys):]+keys[:r%len(keys)]:
                stat,out=measure(functions[name],repeat=1,warmup=0)
                entry=row['variants'].setdefault(name,{'seconds':[],'peak_increment_mib':0.})
                entry['seconds']+=stat['seconds']
                entry['peak_increment_mib']=max(entry['peak_increment_mib'],stat['peak_increment_mib'])
                del out
            print('REPEAT '+str(r)+' '+json.dumps({k:round(v['seconds'][-1],4) for k,v in row['variants'].items()}),flush=True)
        for v in row['variants'].values():v['median_seconds']=float(np.median(v['seconds']))
        # Moving-frame reference is evaluated at moved coordinates, never against old DFT labels.
        _,moved_reference=measure(lambda:original(model,perturbed,chunk),repeat=1,warmup=0)
        moving_functions={
            'moved_rebuild':lambda:flat_density(model,core,perturbed,None,chunk,replay),
            'moved_skin_reuse':lambda:flat_density(model,core,perturbed,cache.get(perturbed),chunk,replay),
        }
        moving={}
        for name,fn in moving_functions.items():
            out=fn();moving[name]={'agreement':check(out,moved_reference),'seconds':[]};del out
        for r in range(args.repeat):
            for name in (list(moving_functions) if r%2==0 else list(reversed(moving_functions))):
                stat,out=measure(moving_functions[name],repeat=1,warmup=0)
                moving[name]['seconds']+=stat['seconds'];del out
        for v in moving.values():v['median_seconds']=float(np.median(v['seconds']))
        row['moving_frame']={'max_atom_displacement':float(displacement.norm(dim=1).max()),
                             'skin':cache.skin,'measurements':moving}
        if sid==args.samples[0]:
            row['profile_synchronized_diagnostic']=profile(model,core,sample,graph,chunk)
        result['records'].append(row)
        save(ROOT/'result.json',result)
        print('DONE '+str(sid)+' '+json.dumps({k:round(v['median_seconds'],4) for k,v in row['variants'].items()}),flush=True)
        del reference,moved_reference,replay,graph,cache,sample,perturbed,candidate,functions,moving_functions
        gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--samples',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--chunk',type=int,default=4096)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,provenance=official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval()
    core=FlatProbe(model).eval()
    result={'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha256(CHECKPOINT),
            'provenance':provenance,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'settings':vars(args),'records':[]}
    if not args.pilot:
        study(model,core,result,args)
        return
    for sid in ([0] if args.pilot else args.samples):
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        print(f'SAMPLE {sid}',flush=True)
        baseline,ref=measure(lambda:original(model,sample),repeat=1 if args.pilot else args.repeat)
        print('ORIGINAL '+json.dumps(baseline),flush=True)
        graph_stat,graph=measure(lambda:ProbeEdges.build(sample),repeat=3)
        classic_stat,classic=measure(lambda:PeriodicQueryGraph(sample['positions'],sample['probes'],sample['cell'],4.),repeat=1)
        # Set equality including all periodic images; retain multiplicity.
        source,receiver,shifts=classic.select(torch.arange(len(sample['probes']),device='cuda'))
        def keys(a,b,s):
            return sorted(map(tuple,torch.cat((a[:,None],b[:,None],s.long()),1).cpu().numpy().tolist()))
        # Avoid millions of Python tuples: lexicographic sort in NumPy instead.
        def sorted_rows(a,b,s):
            x=torch.cat((b[:,None],a[:,None],s.long()),1).cpu().numpy()
            return x[np.lexsort(tuple(x[:,k] for k in range(4,-1,-1)))]
        exact_edges=np.array_equal(sorted_rows(source,receiver,shifts),sorted_rows(graph.source,graph.receiver,graph.shifts))
        if not exact_edges: raise AssertionError('Local stencil and original edge multisets differ')
        del source,receiver,shifts
        row={'id':sid,'metadata':sample['metadata'],'original_2048':baseline,
             'local_graph':graph_stat,'classic_graph':classic_stat,'exact_edge_multiset':exact_edges,
             'edges':len(graph.source),'variants':{}}
        chunks=[2048,4096,8192,16384] if args.pilot else [args.chunk]
        for chunk in chunks:
            for name,fn in [
                (f'official_{chunk}_cached',lambda:original(model,sample,chunk,classic)),
                (f'flat_{chunk}_cached',lambda:flat_density(model,core,sample,graph,chunk)),
                ]:
                stat,out=measure(fn,repeat=2 if args.pilot else args.repeat)
                stat.update(errors(out,ref,sample,sid))
                row['variants'][name]=stat
                print(name+' '+json.dumps(stat),flush=True)
                del out
        chunk=args.chunk
        max_edges=max(int(graph.offsets[min(i+chunk,graph.size)]-graph.offsets[i]) for i in range(0,graph.size,chunk))
        torch.cuda.synchronize(); t=time.perf_counter()
        replay=ProbeReplay(core,chunk,math_ceil(max_edges,1024),64)
        torch.cuda.synchronize()
        row['capture_seconds']=time.perf_counter()-t
        row['edge_capacity']=replay.edge_capacity
        for name,fn in [
            ('flat_fresh',lambda:flat_density(model,core,sample,None,chunk)),
            ('replay_cached_geometry',lambda:flat_density(model,core,sample,graph,chunk,replay)),
            ('replay_fresh_geometry',lambda:flat_density(model,core,sample,None,chunk,replay)),
        ]:
            stat,out=measure(fn,repeat=2 if args.pilot else args.repeat)
            stat.update(errors(out,ref,sample,sid))
            row['variants'][name]=stat
            print(name+' '+json.dumps(stat),flush=True)
            del out
        result['records'].append(row)
        save(ROOT/('pilot.json' if args.pilot else 'result.json'),result)
        del replay,graph,classic,sample,ref
        gc.collect();torch.cuda.empty_cache()


def math_ceil(n,m):
    return ((n+m-1)//m)*m


if __name__=='__main__':
    main()
