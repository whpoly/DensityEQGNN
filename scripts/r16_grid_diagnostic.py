"""Exact atom-centered cutoff edges on a uniform periodic FFT grid.

Borrow Eden's local stencil enumeration, then filter using the true atom
position and R16's strict cutoff. Periodic duplicates are retained. This is
an inference diagnostic, separate from training and the production builder.
"""
import numpy as np
import torch
from density_eqgnn.periodic import CutoffEdges


def uniform_grid_edges(positions,cell,shape,cutoff=3.,atom_chunk=8):
    shape=np.asarray(shape,dtype=np.int64)
    if shape.shape!=(3,) or np.any(shape<=0) or cutoff<=0:
        raise ValueError('Positive three-dimensional grid shape and cutoff required')
    device=positions.device
    grid_numpy=cell.detach().cpu().double().numpy()/shape[:,None]
    # Rounding the atom to the grid shifts its center by no more than this.
    radius=cutoff+.5*np.linalg.norm(grid_numpy,axis=1).sum()
    bounds=np.ceil(radius*np.linalg.norm(np.linalg.inv(grid_numpy),axis=0)).astype(int)
    offsets=np.stack(np.meshgrid(*[np.arange(-b,b+1) for b in bounds],indexing='ij'),-1).reshape(-1,3)
    vectors=offsets@grid_numpy
    offsets=offsets[np.linalg.norm(vectors,axis=1)<=radius+1e-12]
    offset=torch.as_tensor(offsets,device=device,dtype=torch.long)
    dimensions=torch.as_tensor(shape,device=device,dtype=torch.long)
    grid=cell.double()/dimensions[:,None]
    nearest=torch.round(positions.double()@torch.linalg.inv(cell.double())*dimensions).long()
    centers=[]
    neighbors=[]
    displacements=[]
    for start in range(0,len(positions),atom_chunk):
        stop=min(start+atom_chunk,len(positions))
        integer=nearest[start:stop,None]+offset[None]
        delta=integer.double()@grid-positions[start:stop,None].double()
        valid=delta.square().sum(-1)<cutoff**2
        which,within=torch.nonzero(valid,as_tuple=True)
        wrapped=integer[which,within]%dimensions
        centers.append((wrapped[:,0]*dimensions[1]+wrapped[:,1])*dimensions[2]+wrapped[:,2])
        neighbors.append(which+start)
        displacements.append(-delta[which,within])
    center=torch.cat(centers)
    return CutoffEdges(batch=torch.zeros_like(center),center=center,neighbor=torch.cat(neighbors),
                       displacement=torch.cat(displacements))
