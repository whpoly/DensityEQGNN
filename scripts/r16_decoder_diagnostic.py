"""Algebraic decoder diagnostic; does not replace the production implementation.

For regular whitened Gaussian functions, move W_l onto atom coefficients and
absorb scale**(-l) there. Gaussian envelopes are then evaluated once per edge.
All transforms, harmonics and contractions remain float64.
"""
import numpy as np
import torch
from e3nn import o3
from density_eqgnn.radial_basis import radial_transform


def transformed_coefficients(coeff,basis):
    if not (basis.regular_radial and basis.radial_whitening):
        raise ValueError('This diagnostic requires the current regular whitened basis')
    spec=basis.radial_spec
    scales=torch.as_tensor(np.geomspace(spec.minimum,spec.cutoff,spec.num_radial),
                           device=coeff.device,dtype=torch.float64)
    shaped=coeff.double().reshape(*coeff.shape[:2],basis.num_radial,basis.angular_channels)
    blocks=[]
    for degree in range(basis.lmax+1):
        transform=torch.as_tensor(radial_transform(spec,degree)[0],device=coeff.device,dtype=torch.float64)
        block=torch.einsum('kn,banm->bakm',transform,shaped[...,degree**2:(degree+1)**2])
        blocks.append(block*scales[None,None,:,None].pow(-degree))
    return torch.cat(blocks,dim=-1),scales


def decode(coeff,displacement,batch,atoms,targets,num_targets,basis,chunk=65536):
    transformed,scales=transformed_coefficients(coeff,basis)
    density=torch.zeros(num_targets,dtype=torch.float64,device=coeff.device)
    for start in range(0,len(displacement),chunk):
        stop=start+chunk
        disp=displacement[start:stop].double()
        radius=torch.linalg.vector_norm(disp,dim=-1)
        angular=o3.spherical_harmonics(list(range(basis.lmax+1)),disp,normalize=False,normalization='integral')
        radial=torch.exp(-.5*(radius[:,None]/scales).square())
        envelope=.5*(1+torch.cos(torch.pi*radius/basis.cutoff))*(radius<basis.cutoff)
        radial=radial*envelope[:,None]
        selected=transformed[batch[start:stop],atoms[start:stop]]
        value=torch.einsum('enm,en,em->e',selected,radial,angular)
        density=density.index_add(0,targets[start:stop],value)
    return density.to(coeff.dtype)
