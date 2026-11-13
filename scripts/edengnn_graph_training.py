"""Optional exact CUDA-graph replay of the official encoder during training.

Each fixed structure has an independent graph pool. No model, optimizer,
sampling, loss or precision changes. Evaluation and benchmark stay eager.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
from torch_geometric.data import Data

from scripts.nmc_edengnn import EdenDensity, ROOT, prepare_samples, write_json

PREPARED_DATA=[]


class TensorEncoder(torch.nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder

    def forward(self,positions,numbers,edges,shifts):
        data=Data(pos=positions,z=numbers,edge_index=edges,nbr_shift=shifts)
        self.encoder(data)
        return data["node_features"]


def install():
    original=EdenDensity.encode
    def encode(self,data):
        if not self.training or not torch.is_grad_enabled():
            return original(self,data)
        if not hasattr(self,"_encoder_graph_cache"):
            # Plain dictionary intentionally avoids adding duplicate state_dict keys.
            self._encoder_graph_cache={}
        inputs=(data.pos,data.z,data.edge_index,data.nbr_shift)
        key=tuple(x.data_ptr() for x in inputs)
        if key not in self._encoder_graph_cache:
            # Upstream stores this constant as an ordinary CPU attribute.
            # Move the identical index values once so capture has no host copy.
            self.encoder.spharm_edges.coord_change=self.encoder.spharm_edges.coord_change.to(data.pos.device)
            # Capture all structures before the first training autograd graph exists.
            for datum in PREPARED_DATA or [data]:
                tensors=(datum.pos,datum.z,datum.edge_index,datum.nbr_shift)
                cache_key=tuple(x.data_ptr() for x in tensors)
                if cache_key in self._encoder_graph_cache:
                    continue
                wrapper=TensorEncoder(self.encoder)
                captured=torch.cuda.make_graphed_callables(wrapper,tensors,num_warmup_iters=3)
                self._encoder_graph_cache[cache_key]=captured
                print(f"ENCODER_GRAPH_CAPTURE atoms={len(datum.pos)} edges={datum.edge_index.shape[1]} allocated_mib={torch.cuda.memory_allocated()/2**20:.1f}",flush=True)
        features=self._encoder_graph_cache[key](*inputs)
        output=data.clone()
        output["node_features"]=features
        output["node_attrs"]=torch.nn.functional.one_hot(data.z,num_classes=self.encoder.one_hot.num_types).to(data.pos.dtype)
        self.density.conv_to_output_node(output)
        return self.density.reshape_irreps(output["node_charge"]),self.density.z_net(output["node_attrs"]),output
    EdenDensity.encode=encode
    return original


def check():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    model=EdenDensity().cuda()
    saved=torch.load(ROOT/"eden_eight/best.pt",map_location="cuda",weights_only=False)
    model.load_state_dict(saved["model"])
    del saved
    samples=prepare_samples(range(8))
    PREPARED_DATA.extend(s['atom_data'] for s in samples)
    original=install()
    replay=EdenDensity.encode
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    rows=[]
    for iteration in range(16):
        sample=samples[iteration%8]
        ids=sample["split"]["train"][iteration:iteration+1024]
        model.train()
        EdenDensity.encode=original
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started=time.perf_counter()
        reference=model.sampled(sample,ids)
        loss=((reference.double()-sample["target"][ids])/sample["normalizer"]).abs().mean()
        loss.backward()
        torch.cuda.synchronize()
        eager_seconds=time.perf_counter()-started
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        reference=reference.detach()
        del loss
        EdenDensity.encode=replay
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started=time.perf_counter()
        output=model.sampled(sample,ids)
        loss=((output.double()-sample["target"][ids])/sample["normalizer"]).abs().mean()
        loss.backward()
        torch.cuda.synchronize()
        graph_seconds=time.perf_counter()-started
        numerator=denominator=0.
        for name,p in model.named_parameters():
            if name in gradients:
                numerator+=float((p.grad-gradients[name]).double().square().sum())
                denominator+=float(gradients[name].double().square().sum())
        relative=(numerator/max(denominator,1e-30))**.5
        maximum=float((reference-output).detach().abs().max())
        if relative>2e-5 or maximum>2e-5:
            raise RuntimeError(f"CUDA graph mismatch: {relative} / {maximum}")
        rows.append({"iteration":iteration,"sample_id":sample["id"],"max_abs":maximum,
                     "gradient_relative_l2":relative,"eager_seconds":eager_seconds,"graph_seconds":graph_seconds})
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        print(json.dumps(rows[-1]),flush=True)
        del output,loss
    write_json(ROOT/"cuda_graph_audit.json",{"passed":True,"records":rows,
              "note":"All eight structures twice, including replay after optimizer updates; train points only."})


if __name__=="__main__":
    if "--check" in sys.argv:
        check()
    else:
        if not (ROOT/"cuda_graph_audit.json").exists():
            raise RuntimeError("CUDA graph audit required")
        install()
        from scripts import train_nmc_edengnn as training
        original_prepare=training.prepare_samples
        def prepare(*args,**kwargs):
            samples=original_prepare(*args,**kwargs)
            PREPARED_DATA.extend(s['atom_data'] for s in samples)
            return samples
        training.prepare_samples=prepare
        training.main()
