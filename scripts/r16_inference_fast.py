"""Inference optimizations retaining FP64 basis arithmetic and the fixed model."""
import numpy as np
import torch
from e3nn import o3
from ase.data import atomic_numbers

from scripts.r16_decoder_diagnostic import transformed_coefficients


def decode_atomwise(coeff, displacement, batch, atoms, targets, num_targets, basis, chunk=65536):
    """Group edges by atom: (points x radial) @ (radial x angular).

    Avoid materializing a 16x25 coefficient tensor for every edge. Accept
    arbitrary edge ordering and batch indices; the local grid path is already
    ordered, so it needs no sorting. Periodic duplicate edges are retained.
    """
    transformed,scales=transformed_coefficients(coeff,basis)
    flat=transformed.flatten(0,1)
    keys=batch*coeff.shape[1]+atoms
    if len(keys)>1 and bool(torch.any(keys[1:]<keys[:-1])):
        order=torch.argsort(keys,stable=True)
        keys=keys[order]
        displacement,targets=displacement[order],targets[order]
    counts=torch.bincount(keys,minlength=len(flat)).cpu().tolist()
    density=coeff.new_zeros(num_targets,dtype=torch.float64)
    start=0
    for atom,count in enumerate(counts):
        for offset in range(start,start+count,chunk):
            stop=min(offset+chunk,start+count)
            disp=displacement[offset:stop].double()
            radius=torch.linalg.vector_norm(disp,dim=-1)
            angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,
                                           normalize=False,normalization='integral')
            radial=torch.exp(-.5*(radius[:,None]/scales).square())
            envelope=.5*(1+torch.cos(torch.pi*radius/basis.cutoff))*(radius<basis.cutoff)
            value=((radial@flat[atom])*angular).sum(-1)*envelope
            density=density.index_add(0,targets[offset:stop],value)
        start+=count
    return density.to(coeff.dtype)


def decode_grouped(coeff, displacement, batch, atoms, targets, num_targets, basis,
                   atoms_per_chunk=8,point_chunk=65536):
    """Batched atom matrix products with masked padding for unequal edge counts."""
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
        maximum=max(counts_cpu[a:b])
        for start in range(0,maximum,point_chunk):
            local=torch.arange(start,min(start+point_chunk,maximum),device=coeff.device)
            valid=local[None]<counts[a:b,None]
            indices=torch.where(valid,starts[a:b,None]+local[None],0)
            disp=displacement[indices].double()
            radius=torch.linalg.vector_norm(disp,dim=-1)
            angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,
                                           normalize=False,normalization='integral')
            radial=torch.exp(-.5*(radius[...,None]/scales).square())
            envelope=.5*(1+torch.cos(torch.pi*radius/basis.cutoff))*(radius<basis.cutoff)
            value=((radial@flat[a:b])*angular).sum(-1)*envelope*valid
            density=density.index_add(0,targets[indices].flatten(),value.flatten())
    return density.to(coeff.dtype)


class TorchSAD:
    """Same natural cubic POTCAR spline and Fourier sum as make_sad, on GPU.

    Only immutable POTCAR tables are kept; geometry and density are regenerated
    on every invocation. Internal spline, phase and FFT arithmetic is FP64.
    """
    def __init__(self,factors,device='cuda'):
        self.factors={atomic_numbers[s]:(f.reciprocal_cutoff,f.zval,
                     torch.tensor(f.spline.c,dtype=torch.float64,device=device)) for s,f in factors.items()}

    def __call__(self,sample):
        positions=sample['positions'].double()
        cell=sample['cell'].double()
        device=cell.device
        shape=tuple(sample['metadata']['grid_shape'])
        inverse=torch.linalg.inv(cell)
        fractional=positions@inverse
        frequencies=[torch.fft.fftfreq(n,device=device,dtype=torch.float64)*n for n in shape]
        reciprocal=2*torch.pi*inverse.T
        h0,h1,h2=torch.meshgrid(*frequencies,indexing='ij')
        norm=sum((h0*reciprocal[0,k]+h1*reciprocal[1,k]+h2*reciprocal[2,k]).square() for k in range(3)).sqrt()
        fourier=torch.zeros(shape,dtype=torch.complex128,device=device)
        expected=0.
        for number in torch.unique(sample['numbers']).cpu().tolist():
            cutoff,zval,c=self.factors[number]
            selected=fractional[sample['numbers']==number]
            expected+=len(selected)*zval
            coordinate=norm*(200./cutoff)
            clipped=coordinate.clamp(max=999.)
            index=clipped.floor().long().clamp(max=998)
            delta=clipped-index
            form=(((c[0,index]*delta+c[1,index])*delta+c[2,index])*delta+c[3,index])
            form=form*(coordinate<=999.)
            phase=[torch.exp((-2j*torch.pi)*axis[:,None]*selected[None,:,k]) for k,axis in enumerate(frequencies)]
            structure=torch.einsum('ia,ja,ka->ijk',*phase)
            fourier=fourier+form*structure
        raw=torch.fft.ifftn(fourier*np.prod(shape)).real
        if abs(float(raw.mean())-expected)>2e-7:
            raise ValueError('GPU SAD electron count mismatch')
        return (raw.flatten()/sample['metadata']['volume']).float()
