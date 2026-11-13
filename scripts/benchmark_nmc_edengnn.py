"""Serial, synchronized full-grid benchmark of the trained EdenGNN and R16.

Fresh-input timings rebuild geometry and include SAD for Eden. Warm timings
explicitly reuse geometry. Diagnostic sub-stage timings are reported separately
from the uninstrumented whole-model measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import platform
import subprocess
import time
import types

import numpy as np
import torch
from e3nn import o3

from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.compact_basis import CompactBasisConfig, reconstruct_compact_density, reconstruct_compact_from_edges
from density_eqgnn.periodic import cutoff_edges
from density_eqgnn.radial_basis import radial_values
from scripts.compare_local_nmc_oracles import load_sample
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.nmc_edengnn import ROOT,EdenDensity,EdenStencil,atom_data,make_sad,write_json
from scripts.nmc_probe_holdout import DEFAULT_ROOT,save_split,sha256,PeriodicQueryGraph,compact_prediction
from scripts.r16_decoder_diagnostic import decode as reordered_decode

R16=DEFAULT_ROOT/"r16_eight/best.pt"


def timed(function,repeat=5,warmup=1):
    for _ in range(warmup):
        warm=function()
        del warm
    torch.cuda.synchronize()
    values=[]
    peak_absolute=peak_increment=0
    output=None
    for _ in range(repeat):
        del output
        torch.cuda.synchronize()
        base=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        output=function()
        torch.cuda.synchronize()
        values.append(time.perf_counter()-start)
        peak=torch.cuda.max_memory_allocated()
        peak_absolute=max(peak_absolute,peak)
        peak_increment=max(peak_increment,peak-base)
    return {"seconds":values,"median_seconds":float(np.median(values)),
            "p90_seconds":float(np.percentile(values,90)),
            "peak_allocated_mib":peak_absolute/2**20,
            "peak_increment_mib":peak_increment/2**20},output


def r16_inputs(sample):
    return (sample["numbers"][None],sample["positions"].float()[None],
            torch.ones(1,len(sample["numbers"]),dtype=torch.bool,device="cuda"),
            sample["cell"].float()[None])


def r16_edges(sample,basis):
    return cutoff_edges(sample["probes"][None],sample["positions"][None],
          torch.ones(1,len(sample["probes"]),dtype=torch.bool,device="cuda"),
          torch.ones(1,len(sample["numbers"]),dtype=torch.bool,device="cuda"),
          sample["cell"][None],basis.cutoff,True,chunk_size=4096)


def r16_decode(coeff,sample,edges,basis):
    return reconstruct_compact_from_edges(coeff,-edges.displacement,edges.batch,edges.neighbor,
                                         edges.center,len(sample["probes"]),basis)


def fresh_r16(model,sample,basis):
    args=r16_inputs(sample)
    coeff=model(*args)["coefficients"]
    return reconstruct_compact_density(coeff,sample["positions"][None],sample["probes"][None],
        args[2],torch.ones(1,len(sample["probes"]),device="cuda",dtype=torch.bool),sample["cell"][None],basis)[0]


def prepare_native_eden(sample,factors):
    prepared=dict(sample)
    prepared["atom_data"]=atom_data(sample)
    prepared["stencil"]=EdenStencil(sample,csr=False)
    prepared["native_data"]=prepared["atom_data"].clone()
    for k,v in prepared["stencil"].native_fields().items():
        prepared["native_data"][k]=v
    prepared["sad"]=make_sad(sample,factors)
    return prepared


def r16_parts(model,sample,geometry):
    """Stage the original forward algebra without changing operations."""
    numbers,_,mask,_=r16_inputs(sample)
    distances,directions,pair_mask,sources=geometry
    radial=model.edge_basis(distances)
    harmonics=o3.spherical_harmonics(list(range(model.config.resolved_hidden_lmax+1)),
                                    directions,normalize=True,normalization="integral")
    features=model.embedding(numbers)
    for interaction in model.interactions:
        features=interaction(features,harmonics,radial,pair_mask,sources)
        features*=mask[...,None].to(features.dtype)
    return features,radial,directions,pair_mask,sources


def r16_head(model,sample,state):
    features,radial,directions,pair_mask,sources=state
    harmonics=o3.spherical_harmonics(list(range(model.config.resolved_readout_edge_lmax+1)),
                                    directions,normalize=True,normalization="integral")
    degree=model.readout(features,harmonics,radial,pair_mask,sources)
    blocks=model._degree_major_to_radial_major(degree)
    numbers=sample["numbers"][None]
    baseline=(model.atomic_scalar_baseline[numbers] if model.learned_scalar_baseline is None
              else model.learned_scalar_baseline(numbers))
    blocks[...,0]+=baseline
    return blocks.reshape(1,len(sample["numbers"]),-1)


def diagnostic_eden_decode(model,sample):
    """Synchronized decomposition of the same native density-layer algebra."""
    accumulated=defaultdict(float)
    def part(name,fn):
        torch.cuda.synchronize()
        start=time.perf_counter()
        value=fn()
        torch.cuda.synchronize()
        accumulated[name]+=time.perf_counter()-start
        return value
    d=sample["native_data"].clone()
    part("encoder",lambda:model.encoder(d))
    part("coefficient_head",lambda:model.density.conv_to_output_node(d))
    layer=model.density
    h=layer.reshape_irreps(d["node_charge"])
    z=part("element_embedding",lambda:layer.z_net(d["node_attrs"]))
    pos_res=d["pos"]-d["pos_n"].float()@d["grid"]
    pe=part("grid_displacements",lambda:(d["edge_vec_probes"][None]-pos_res[:,None])[:,:,[1,2,0]])
    pl=part("grid_distances",lambda:torch.linalg.vector_norm(pe,dim=-1))
    atom_chunk=max(1,layer.chunk_size_predict//d["npb"])
    pieces=[]
    for start in range(0,d["nat"],atom_chunk):
        stop=min(start+atom_chunk,d["nat"])
        angular=part("spherical_harmonics",lambda:layer.spharm_edges_probe(pe[start:stop]))
        radial=part("radial_embedding",lambda:torch.cat((layer.radial_basis_probe(pl[start:stop]),
                    z[start:stop,None].expand(-1,d["npb"],-1)),dim=-1))
        filters=part("radial_mlp",lambda:layer.edge_net(radial).reshape(stop-start,d["npb"],layer.n_channels,layer.l_max+1))
        values=part("contraction",lambda:torch.einsum("ipl,ipl->ip",
                (h[start:stop,None]*filters[...,layer.l_index]).sum(dim=-2),angular))
        pieces.append(values)
    p=part("chunk_concatenation",lambda:torch.cat(pieces,dim=0))
    def scatter():
        grid=torch.zeros((*d["grid_shape"].tolist(),1),device="cuda")
        indices=((d["pos_n"][:,None]+d["map_probe"][None])%d["grid_shape"]).reshape(-1,3)
        grid.index_put_(tuple(indices.T),p.reshape(-1,1),accumulate=True)
        return grid.flatten()+sample["sad"]
    result=part("grid_accumulation_and_sad",scatter)
    return dict(accumulated),result


def diagnostic_r16_decode(coeff,sample,edges,basis):
    """CUDA-event attribution; instrumentation is excluded from headline timings."""
    events=[]
    def part(name,fn):
        before=torch.cuda.Event(enable_timing=True)
        after=torch.cuda.Event(enable_timing=True)
        before.record()
        value=fn()
        after.record()
        events.append((name,before,after))
        return value
    torch.cuda.synchronize()
    started=time.perf_counter()
    shaped=coeff.reshape(1,len(sample["numbers"]),basis.num_radial,basis.angular_channels)
    density=torch.zeros(len(sample["probes"]),device="cuda",dtype=torch.float64)
    for start in range(0,len(edges.center),8192):
        stop=start+8192
        displacement=-edges.displacement[start:stop]
        radius=part("distances",lambda:torch.linalg.vector_norm(displacement,dim=-1))
        angular=part("spherical_harmonics",lambda:o3.spherical_harmonics(list(range(basis.lmax+1)),
                     displacement,normalize=False,normalization="integral"))
        selected=part("coefficient_gather_cast",lambda:shaped[edges.batch[start:stop],edges.neighbor[start:stop]].double())
        values=radius*0
        for degree in range(basis.lmax+1):
            radial=part("radial_functions_and_whitening",lambda:radial_values(radius,basis.radial_spec,degree,
                        whiten=True,solid_harmonics=True))
            contribution=part("contraction",lambda:torch.einsum("enm,en,em->e",
                selected[:,:,degree**2:(degree+1)**2],radial,angular[:,degree**2:(degree+1)**2]))
            values=values+contribution
        density=part("grid_accumulation",lambda:density.index_add(0,edges.center[start:stop],values))
    torch.cuda.synchronize()
    totals=defaultdict(float)
    calls=defaultdict(int)
    for name,before,after in events:
        totals[name]+=before.elapsed_time(after)/1000
        calls[name]+=1
    return {"cuda_event_seconds":dict(totals),"calls":dict(calls),
            "instrumented_wall_seconds":time.perf_counter()-started},density.to(coeff.dtype)


def metrics(pred,sample,split):
    row={}
    for name,ids in split.items():
        if name=="selection":
            continue
        target=sample["target"][ids]
        delta=pred[ids].double()-target
        row[name+"_nmae"]=float(delta.abs().sum()/target.abs().sum())
        row[name+"_charge_error"]=float(delta.sum()/target.abs().sum())
    row["full_grid_nmae"]=float((pred.double()-sample["target"]).abs().sum()/sample["target"].abs().sum())
    row["negative_fraction"]=float((-pred.double()).clamp_min(0).sum()/sample["target"].abs().sum())
    return row


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--repeat",type=int,default=5)
    parser.add_argument("--output",type=Path,default=ROOT/"benchmark.json")
    args=parser.parse_args()
    if not args.smoke and not (ROOT/"eden_eight/selection_frozen.json").exists():
        raise RuntimeError("Training and validation selection must finish before benchmark")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    tick=time.perf_counter()
    eden=EdenDensity().cuda().eval()
    state=torch.load(ROOT/"eden_eight/best.pt",map_location="cuda",weights_only=False)
    eden.load_state_dict(state["model"])
    torch.cuda.synchronize()
    eden_load=time.perf_counter()-tick
    tick=time.perf_counter()
    rstate=torch.load(R16,map_location="cuda",weights_only=False)
    r16=build_compact_coefficient_model(rstate["model_config"]).cuda().eval()
    r16.load_state_dict(rstate["model"])
    basis=CompactBasisConfig(**rstate["basis_config"])
    torch.cuda.synchronize()
    r16_load=time.perf_counter()-tick
    del state,rstate
    if not args.smoke:
        frozen=json.loads((ROOT/"eden_eight/selection_frozen.json").read_text())
        if sha256(ROOT/"eden_eight/best.pt")!=frozen["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint changed after freezing")
    factors=read_form_factors(Path("potpaw54.zip"))
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"smoke_only":args.smoke,
            "hardware":{"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
                        "platform":platform.platform(),"cpu":platform.processor(),"torch_threads":2},
            "checkpoint_sha256":{"eden":sha256(ROOT/"eden_eight/best.pt"),"r16":sha256(R16)},
            "model_load_seconds":{"eden":eden_load,"r16":r16_load},
            "repeats":args.repeat,"warmup":1,"records":[],
            "notes":["fresh time includes geometry and Eden POTCAR SAD; model/checkpoint and POTCAR loading excluded",
                     "dataset_read includes target CHGCAR and is diagnostic only, not an inference input requirement",
                     "warm timings reuse geometry; no cached learned outputs",
                     "native R16 decoder float64, Eden decoder float32; network float32, TF32 off",
                     "peak allocation includes both resident models; incremental peak is separately reported",
                     "diagnostic decoder stages use additional synchronization; use uninstrumented totals for speed comparison"]}
    result["nvidia_smi_before"]=subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout
    ids=[0] if args.smoke else range(8)
    for sid in ids:
        print(f"BENCHMARK sample={sid} starting",flush=True)
        path=Path(f"data/nmc_cache/{sid}.CHGCAR.lz4")
        read_time,cpu=timed(lambda:load_sample(path,"cpu"),args.repeat,0)
        def transfer():
            return {k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
        transfer_time,sample=timed(transfer,args.repeat,0)
        row={"sample_id":sid,"formula":sample["metadata"]["formula"],"atoms":len(sample["numbers"]),
             "grid_points":len(sample["probes"]),"dataset_read":read_time,"host_to_device":transfer_time}
        # Alternate model order between structures; neither model trains here.
        fresh={}
        for name in (("eden","r16") if sid%2==0 else ("r16","eden")):
            call=(lambda:eden.native(prepare_native_eden(sample,factors))) if name=="eden" else (lambda:fresh_r16(r16,sample,basis))
            fresh[name],prediction=timed(call,args.repeat)
            print(f"FRESH sample={sid} model={name} seconds={fresh[name]['median_seconds']:.4f}",flush=True)
            if not args.smoke:
                split,_=save_split(DEFAULT_ROOT/"spatial_splits",len(sample["target"]),sid,sample["metadata"]["sha256"])
                split={k:torch.tensor(v,device="cuda") for k,v in split.items()}
                row[name+"_accuracy"]=metrics(prediction,sample,split)
            if name=="eden":
                eden_reference=prediction
            else:
                r16_reference=prediction
        row["fresh"]=fresh
        # Geometry and prior preparation are explicitly charged in fresh timings.
        stages={}
        stages["eden_atom_graph"],ad=timed(lambda:atom_data(sample),args.repeat)
        stages["eden_grid_geometry"],stencil=timed(lambda:EdenStencil(sample,csr=False),args.repeat)
        stages["eden_sad"],sad=timed(lambda:make_sad(sample,factors),args.repeat)
        prepared=dict(sample,atom_data=ad,stencil=stencil,sad=sad,native_data=ad.clone())
        for key,value in stencil.native_fields().items():
            prepared["native_data"][key]=value
        stages["eden_encoder"],encoded=timed(lambda:encode_only(eden,ad),args.repeat)
        stages["eden_coefficient_head"],_=timed(lambda:eden.density.conv_to_output_node(encoded.clone()),args.repeat)
        def eden_decode():
            d=encoded.clone()
            for key,value in stencil.native_fields().items():
                d[key]=value
            eden.density(d)
            return d["charge"].flatten()+sad
        stages["eden_decoder_including_head"],eden_decoded=timed(eden_decode,args.repeat)
        stages["eden_warm_total"],_=timed(lambda:eden.native(prepared),args.repeat)
        torch.testing.assert_close(eden_decoded,eden_reference,rtol=2e-5,atol=3e-5)
        rargs=r16_inputs(sample)
        stages["r16_atom_graph"],geometry=timed(lambda:r16._geometry(rargs[1],rargs[2],rargs[3]),args.repeat)
        stages["r16_encoder"],rrep=timed(lambda:r16_parts(r16,sample,geometry),args.repeat)
        stages["r16_coefficient_head"],coeff=timed(lambda:r16_head(r16,sample,rrep),args.repeat)
        original_geometry=r16._geometry
        r16._geometry=types.MethodType(lambda self,*unused:geometry,r16)
        stages["r16_network_including_head"],reference_coeff=timed(lambda:r16(*rargs)["coefficients"],args.repeat)
        torch.testing.assert_close(coeff,reference_coeff,rtol=1e-5,atol=1e-6)
        stages["r16_grid_geometry_production"],edges=timed(lambda:r16_edges(sample,basis),args.repeat)
        stages["r16_decoder"],rdecoded=timed(lambda:r16_decode(coeff,sample,edges,basis),args.repeat)
        stages["r16_warm_total"],_=timed(lambda:r16_decode(r16(*rargs)["coefficients"],sample,edges,basis),args.repeat)
        r16._geometry=original_geometry
        torch.testing.assert_close(rdecoded,r16_reference,rtol=1e-5,atol=2e-6)
        stages["r16_grid_geometry_kdtree"],graph=timed(lambda:PeriodicQueryGraph(sample["positions"],sample["probes"],sample["cell"],3.),args.repeat)
        query_sample=dict(sample,graph=graph)
        query_ids=torch.arange(len(sample["probes"]),device="cuda")
        def query_decode():
            return torch.cat([compact_prediction(query_sample,coeff,part,basis) for part in query_ids.split(2048)])
        stages["r16_query_chunk_decoder"],query_pred=timed(query_decode,args.repeat)
        torch.testing.assert_close(query_pred,r16_reference,rtol=2e-5,atol=3e-6)
        row["r16_query_vs_production_max_abs"]=float((query_pred-r16_reference).abs().max())
        row["stages"]=stages
        row["edge_counts"]={"eden_native_atom_grid":stencil.num_atoms*stencil.npb,
                            "r16_atom_grid":len(edges.center),"eden_atom_graph":ad.edge_index.shape[1],
                            "r16_atom_graph":int(geometry[2].sum())}
        diagnostic,diagnostic_prediction=diagnostic_eden_decode(eden,prepared)
        torch.testing.assert_close(diagnostic_prediction,eden_reference,rtol=2e-5,atol=3e-5)
        row["eden_decoder_synchronized_diagnostic_seconds"]=diagnostic
        row["native_eden_diagnostic_max_abs"]=float((diagnostic_prediction-eden_reference).abs().max())
        rdiagnostic,rdiagnostic_prediction=diagnostic_r16_decode(coeff,sample,edges,basis)
        torch.testing.assert_close(rdiagnostic_prediction,r16_reference,rtol=1e-5,atol=3e-6)
        row["r16_decoder_event_diagnostic"]=rdiagnostic
        stages['r16_reordered_decoder'],reordered=timed(lambda:reordered_decode(
            coeff,-edges.displacement,edges.batch,edges.neighbor,edges.center,len(sample['probes']),basis),args.repeat)
        torch.testing.assert_close(reordered,r16_reference,rtol=3e-5,atol=5e-5)
        row['r16_reordered_max_abs']=float((reordered-r16_reference).abs().max())
        row['r16_reordered_relative_l1']=float((reordered-r16_reference).double().abs().sum()/r16_reference.double().abs().sum())
        if not args.smoke:
            row['r16_reordered_accuracy']=metrics(reordered,sample,split)
        def reordered_fresh():
            new_coeff=r16(*r16_inputs(sample))['coefficients']
            new_edges=r16_edges(sample,basis)
            return reordered_decode(new_coeff,-new_edges.displacement,new_edges.batch,new_edges.neighbor,
                                    new_edges.center,len(sample['probes']),basis)
        stages['r16_reordered_fresh_total'],_=timed(reordered_fresh,args.repeat)
        del reordered
        if not args.smoke:
            # Audit sampled adapter against trained native output without selecting weights.
            stages["eden_grid_geometry_csr"],prepared["stencil"]=timed(lambda:EdenStencil(sample,csr=True),args.repeat)
            rep=eden.encode(ad)
            all_ids=torch.arange(len(sample["probes"]),device="cuda")
            def eden_query_decode():
                return torch.cat([eden.sampled(prepared,part,rep) for part in all_ids.split(2048)])
            stages["eden_query_chunk_decoder"],query_eden=timed(eden_query_decode,args.repeat)
            torch.testing.assert_close(query_eden,eden_reference,rtol=3e-5,atol=5e-5)
            row["native_vs_query_full_max_abs"]=float((query_eden-eden_reference).abs().max())
            del query_eden,all_ids,rep
            rep=eden.encode(ad)
            selection=torch.tensor(np.arange(0,len(sample["probes"]),997),device="cuda")
            sparse=eden.sampled(prepared,selection,rep)
            torch.testing.assert_close(sparse,eden_reference[selection],rtol=3e-5,atol=5e-5)
            row["trained_native_vs_sampled_max_abs"]=float((sparse-eden_reference[selection]).abs().max())
        result["records"].append(row)
        write_json(args.output,result)
        print(f"BENCHMARK sample={sid} done",flush=True)
        del prepared,stencil,ad,encoded,geometry,rrep,coeff,reference_coeff,edges,graph,query_sample,query_ids
        del eden_reference,r16_reference,eden_decoded,rdecoded,query_pred,diagnostic_prediction,prediction,sample,cpu
        gc.collect()
        torch.cuda.empty_cache()
    result["nvidia_smi_after"]=subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout
    result["complete"]=True
    write_json(args.output,result)
    print("BENCHMARK_COMPLETE "+str(args.output),flush=True)


def encode_only(model,data):
    data=data.clone()
    model.encoder(data)
    return data


if __name__=="__main__":
    main()
