"""Optional exact-algebra diagnostic: contiguous query segments instead of atomic scatter."""
import torch
from scripts.deepdft_inference_fast import FlatProbe


class SegmentProbe(FlatProbe):
    def aggregate(self, layer, scalar, vector, projection, source, receiver, geometry, size):
        direction,radial,envelope,lengths=geometry
        gates=(layer.filter_layer(radial)*envelope)*projection[source]
        gv,ge,gs=gates.split(scalar.shape[-1],1)
        ms=scalar[source]*gs
        mv=vector[source]*gv[:,None]+ge[:,None]*direction[:,:,None]
        return (torch.segment_reduce(ms,'sum',lengths=lengths,unsafe=True),
                torch.segment_reduce(mv,'sum',lengths=lengths,unsafe=True))

    def forward(self,source,receiver,diff,scalars,vectors,projections,size):
        geometry=(*self.geometry(diff),torch.bincount(receiver,minlength=size))
        s=scalars.new_zeros(size,scalars.shape[-1])
        v=vectors.new_zeros(size,3,scalars.shape[-1])
        for k,(layer,update) in enumerate(zip(self.probe.message_layers,self.probe.scalar_vector_update)):
            a,b=self.aggregate(layer,scalars[k],vectors[k],projections[k],source,receiver,geometry,size)
            gs,gv=layer.update_gate_mlp(a).split(s.shape[-1],1)
            s,v=update(gs*s+(1-gs)*a,gv[:,None]*v+(1-gv[:,None])*b)
        return self.probe.readout_function(s).squeeze(1)


@torch.no_grad()
def main():
    import argparse
    import json
    from pathlib import Path
    from scripts.benchmark_deepdft_reuse import measure,check,ROOT,CHECKPOINT,save
    from scripts.compare_local_nmc_oracles import load_sample
    from scripts.nmc_probe_holdout import official_model,DEFAULT_ROOT
    from scripts.deepdft_inference_fast import ProbeEdges,flat_density
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,nargs='+',default=[0]);args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,_=official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model']);model.eval()
    flat=FlatProbe(model).eval();segment=SegmentProbe(model).eval();result=[]
    for sid in args.samples:
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        graph=ProbeEdges.build(sample)
        a,ref=measure(lambda:flat_density(model,flat,sample,graph,4096),repeat=3)
        b,out=measure(lambda:flat_density(model,segment,sample,graph,4096),repeat=3)
        row={'id':sid,'flat':a,'segment':b,'agreement':check(out,ref)}
        result.append(row);print(json.dumps(row),flush=True)
        save(ROOT/'segment_diagnostic.json',result)


if __name__=='__main__':main()
