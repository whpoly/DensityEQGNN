"""Exact-architecture inference graph replay and direct uniform-grid decoding."""
from collections import OrderedDict
import time

import numpy as np
import torch
from e3nn import o3

from scripts.r16_decoder_diagnostic import transformed_coefficients
from density_eqgnn.radial_basis import radial_transform


class GraphCoefficients:
    """Bounded shape cache for a fixed evaluation model; never cache predictions.

    New input tensor values (including edge indices and shifts) are copied on
    every call. First use of a shape includes warm-up/capture overhead. Return
    a clone so later replays cannot silently change an earlier result. This
    wrapper is single-stream, single-caller inference only.
    """
    fields=('pos','z','edge_index','nbr_shift')

    def __init__(self,model,max_graphs=2):
        if max_graphs<1:
            raise ValueError('max_graphs must be positive')
        self.model=model
        self.max_graphs=max_graphs
        self.cache=OrderedDict()
        self.captures=[]

    @torch.no_grad()
    def __call__(self,data):
        if self.model.training or data.pos.device.type!='cuda':
            raise ValueError('GraphCoefficients requires an evaluation model and CUDA inputs')
        key=tuple((name,tuple(data[name].shape),data[name].dtype,str(data[name].device)) for name in self.fields)
        if key not in self.cache:
            if len(self.cache)>=self.max_graphs:
                torch.cuda.synchronize(data.pos.device)
                self.cache.popitem(last=False)
            self.model.encoder.spharm_edges.coord_change=self.model.encoder.spharm_edges.coord_change.to(data.pos.device)
            torch.cuda.synchronize(data.pos.device)
            tick=time.perf_counter()
            static=data.clone()
            stream=torch.cuda.Stream(device=data.pos.device)
            stream.wait_stream(torch.cuda.current_stream(data.pos.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output=self.model.coefficients(static)
            torch.cuda.current_stream(data.pos.device).wait_stream(stream)
            del output
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output=self.model.coefficients(static)
            torch.cuda.synchronize(data.pos.device)
            self.captures.append({'atoms':len(data.pos),'edges':data.edge_index.shape[1],
                                  'setup_seconds':time.perf_counter()-tick})
            self.cache[key]=(static,graph,output)
        static,graph,output=self.cache.pop(key)
        self.cache[key]=(static,graph,output)
        for name in self.fields:
            static[name].copy_(data[name])
        graph.replay()
        return output.clone()


def decode_uniform(coeff,positions,cell,shape,basis,atoms_per_chunk=8):
    """Generate candidate grid points and decode together, without an edge list.

    All candidates use the same conservative stencil as uniform_grid_edges.
    Outside the exact atom cutoff contributions are zero. All periodic image
    contributions are retained; float64 basis/transformation is unchanged.
    """
    if coeff.shape[0]!=1 or coeff.shape[1]!=len(positions):
        raise ValueError('One structure with matching atom coefficients required')
    shape=np.asarray(shape,dtype=np.int64)
    if shape.shape!=(3,) or np.any(shape<=0) or atoms_per_chunk<1:
        raise ValueError('Positive grid dimensions and atom chunk required')
    grid_numpy=cell.detach().cpu().double().numpy()/shape[:,None]
    radius=basis.cutoff+.5*np.linalg.norm(grid_numpy,axis=1).sum()
    bounds=np.ceil(radius*np.linalg.norm(np.linalg.inv(grid_numpy),axis=0)).astype(int)
    offsets=np.stack(np.meshgrid(*[np.arange(-b,b+1) for b in bounds],indexing='ij'),-1).reshape(-1,3)
    offsets=offsets[np.linalg.norm(offsets@grid_numpy,axis=1)<=radius+1e-12]
    offset=torch.as_tensor(offsets,device=positions.device,dtype=torch.long)
    dimensions=torch.as_tensor(shape,device=positions.device,dtype=torch.long)
    grid=cell.double()/dimensions[:,None]
    nearest=torch.round(positions.double()@torch.linalg.inv(cell.double())*dimensions).long()
    transformed,scales=transformed_coefficients(coeff,basis)
    density=coeff.new_zeros(int(np.prod(shape)),dtype=torch.float64)
    for start in range(0,len(positions),atoms_per_chunk):
        stop=min(start+atoms_per_chunk,len(positions))
        integer=nearest[start:stop,None]+offset[None]
        disp=integer.double()@grid-positions[start:stop,None].double()
        distance=torch.linalg.vector_norm(disp,dim=-1)
        angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,
                                       normalize=False,normalization='integral')
        radial=torch.exp(-.5*(distance[...,None]/scales).square())
        envelope=.5*(1+torch.cos(torch.pi*distance/basis.cutoff))*(distance<basis.cutoff)
        values=((radial@transformed[0,start:stop])*angular).sum(-1)*envelope
        wrapped=integer%dimensions
        targets=(wrapped[...,0]*dimensions[1]+wrapped[...,1])*dimensions[2]+wrapped[...,2]
        density=density.index_add(0,targets.flatten(),values.flatten())
    return density.to(coeff.dtype)


def stencil_offsets(cell,shape,cutoff):
    shape=np.asarray(shape,dtype=np.int64)
    grid=cell.detach().cpu().double().numpy()/shape[:,None]
    radius=cutoff+.5*np.linalg.norm(grid,axis=1).sum()
    bounds=np.ceil(radius*np.linalg.norm(np.linalg.inv(grid),axis=0)).astype(int)
    offsets=np.stack(np.meshgrid(*[np.arange(-b,b+1) for b in bounds],indexing='ij'),-1).reshape(-1,3)
    offsets=offsets[np.linalg.norm(offsets@grid,axis=1)<=radius+1e-12]
    return torch.as_tensor(offsets,device=cell.device,dtype=torch.long)


class UniformDecoderKernel(torch.nn.Module):
    """Capture-safe pure tensor path; this is not a custom fused CUDA kernel."""
    def __init__(self,basis,shape,device,atoms_per_chunk=8):
        super().__init__()
        self.basis=basis
        self.shape=tuple(shape)
        self.atoms_per_chunk=atoms_per_chunk
        self.register_buffer('dimensions',torch.tensor(shape,device=device,dtype=torch.long))
        spec=basis.radial_spec
        scales=torch.tensor(np.geomspace(spec.minimum,spec.cutoff,spec.num_radial),device=device,dtype=torch.float64)
        self.register_buffer('scales',scales)
        for l in range(basis.lmax+1):
            self.register_buffer(f'w{l}',torch.tensor(radial_transform(spec,l)[0],device=device,dtype=torch.float64))
            self.register_buffer(f'scale{l}',scales.pow(-l))

    def forward(self,coeff,positions,cell,offset):
        basis=self.basis
        shaped=coeff.double().reshape(1,len(positions),basis.num_radial,basis.angular_channels)
        blocks=[]
        for l in range(basis.lmax+1):
            block=torch.einsum('kn,banm->bakm',getattr(self,f'w{l}'),shaped[...,l*l:(l+1)**2])
            blocks.append(block*getattr(self,f'scale{l}')[None,None,:,None])
        transformed=torch.cat(blocks,dim=-1)
        grid=cell/self.dimensions[:,None]
        nearest=torch.round(positions@torch.linalg.inv_ex(cell)[0]*self.dimensions).long()
        density=coeff.new_zeros(int(np.prod(self.shape)),dtype=torch.float64)
        for start in range(0,len(positions),self.atoms_per_chunk):
            stop=min(start+self.atoms_per_chunk,len(positions))
            integer=nearest[start:stop,None]+offset[None]
            disp=integer.double()@grid-positions[start:stop,None]
            distance=torch.linalg.vector_norm(disp,dim=-1)
            angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,
                                           normalize=False,normalization='integral')
            radial=torch.exp(-.5*(distance[...,None]/self.scales).square())
            envelope=.5*(1+torch.cos(torch.pi*distance/basis.cutoff))*(distance<basis.cutoff)
            values=((radial@transformed[0,start:stop])*angular).sum(-1)*envelope
            wrapped=integer%self.dimensions
            targets=(wrapped[...,0]*self.dimensions[1]+wrapped[...,1])*self.dimensions[2]+wrapped[...,2]
            density=density.index_add(0,targets.flatten(),values.flatten())
        return density.to(coeff.dtype)


class GraphUniformDecoder:
    """Bounded graph cache with fresh stencil/input copies on every invocation."""
    def __init__(self,basis,max_graphs=2):
        if max_graphs<1:
            raise ValueError('max_graphs must be positive')
        self.basis=basis
        self.max_graphs=max_graphs
        self.cache=OrderedDict()
        self.captures=[]

    @torch.no_grad()
    def __call__(self,coeff,positions,cell,shape):
        shape=tuple(int(n) for n in shape)
        if len(shape)!=3 or any(n<=0 for n in shape) or coeff.shape[0]!=1 or coeff.shape[1]!=len(positions):
            raise ValueError('Positive 3D grid and single-structure coefficients required')
        if coeff.device.type!='cuda':
            raise ValueError('CUDA inputs required')
        positions,cell=positions.double(),cell.double()
        offsets=stencil_offsets(cell,shape,self.basis.cutoff)
        tensors=(coeff,positions,cell,offsets)
        key=(shape,tuple((tuple(x.shape),x.dtype,str(x.device)) for x in tensors))
        if key not in self.cache:
            if len(self.cache)>=self.max_graphs:
                torch.cuda.synchronize(coeff.device)
                self.cache.popitem(last=False)
            torch.cuda.synchronize(coeff.device)
            tick=time.perf_counter()
            kernel=UniformDecoderKernel(self.basis,shape,coeff.device)
            static=tuple(x.clone() for x in tensors)
            stream=torch.cuda.Stream(device=coeff.device)
            stream.wait_stream(torch.cuda.current_stream(coeff.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output=kernel(*static)
            torch.cuda.current_stream(coeff.device).wait_stream(stream)
            del output
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output=kernel(*static)
            torch.cuda.synchronize(coeff.device)
            self.captures.append({'atoms':len(positions),'stencil_size':len(offsets),'shape':shape,
                                  'setup_seconds':time.perf_counter()-tick})
            self.cache[key]=(kernel,static,graph,output)
        kernel,static,graph,output=self.cache.pop(key)
        self.cache[key]=(kernel,static,graph,output)
        for destination,source in zip(static,tensors):
            destination.copy_(source)
        graph.replay()
        return output.clone()


def decode_angular_first(coeff,displacement,batch,atoms,targets,num_targets,basis,
                         atoms_per_chunk=8,point_chunk=65536):
    """Contract 25 angular channels to 16 radial channels before radial sum.

    Same FP64 tensor contraction, but the matrix-product output has 16 rather
    than 25 channels. Release angular intermediates before creating radials.
    """
    transformed,scales=transformed_coefficients(coeff,basis)
    flat=transformed.flatten(0,1)
    keys=batch*coeff.shape[1]+atoms
    if len(keys)>1 and bool(torch.any(keys[1:]<keys[:-1])):
        order=torch.argsort(keys,stable=True)
        keys=keys[order]
        displacement,targets=displacement[order],targets[order]
    counts=torch.bincount(keys,minlength=len(flat))
    counts_cpu=counts.cpu().tolist()
    starts=counts.cumsum(0)-counts
    density=coeff.new_zeros(num_targets,dtype=torch.float64)
    for a in range(0,len(flat),atoms_per_chunk):
        b=min(a+atoms_per_chunk,len(flat))
        for start in range(0,max(counts_cpu[a:b]),point_chunk):
            local=torch.arange(start,min(start+point_chunk,max(counts_cpu[a:b])),device=coeff.device)
            valid=local[None]<counts[a:b,None]
            indices=torch.where(valid,starts[a:b,None]+local[None],0)
            disp=displacement[indices].double()
            angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,
                                           normalize=False,normalization='integral')
            mixed=angular@flat[a:b].transpose(-1,-2)
            del angular
            radius=torch.linalg.vector_norm(disp,dim=-1)
            radial=torch.exp(-.5*(radius[...,None]/scales).square())
            envelope=.5*(1+torch.cos(torch.pi*radius/basis.cutoff))*(radius<basis.cutoff)
            value=(mixed*radial).sum(-1)*envelope*valid
            density=density.index_add(0,targets[indices].flatten(),value.flatten())
    return density.to(coeff.dtype)
