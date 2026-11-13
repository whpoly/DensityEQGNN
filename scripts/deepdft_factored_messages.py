"""Hoist atom-only products out of edges; FP32 algebra, original parameters."""
import torch

from scripts.deepdft_inference_fast import FlatProbe


class FactoredProbe(FlatProbe):
    def atoms(self,representation,capacity=None):
        if capacity is not None:
            raise ValueError('This implementation does not use padded CUDA replay')
        scalars,vectors = representation
        prepared_s,prepared_v,prepared_e = [],[],[]
        for layer,s,v in zip(self.probe.message_layers,scalars,vectors):
            gv,ge,gs = layer.scalar_message_mlp(s).split(s.shape[-1],1)
            prepared_s.append(s*gs)
            prepared_v.append(v*gv[:,None])
            prepared_e.append(ge)
        return torch.stack(prepared_s),torch.stack(prepared_v),torch.stack(prepared_e)

    def aggregate(self,layer,scalar,vector,projection,source,receiver,geometry,size):
        direction,radial,envelope = geometry
        fv,fe,fs = (layer.filter_layer(radial)*envelope).split(scalar.shape[-1],1)
        ms = scalar[source]*fs
        mv = vector[source]*fv[:,None]+(projection[source]*fe)[:,None]*direction[:,:,None]
        a = scalar.new_zeros(size,scalar.shape[-1]).index_add_(0,receiver,ms)
        b = vector.new_zeros(size,3,scalar.shape[-1]).index_add_(0,receiver,mv)
        return a,b
