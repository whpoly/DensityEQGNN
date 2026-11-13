"""Exact linear parameterization of the existing maximum-order MACE readout.

No atom-specific parameters are introduced. Each column represents an existing
direct weight, last radial-MLP weight/bias, or active element baseline. The
nonlinear trunk and the first two radial-MLP layers remain frozen.
"""
from __future__ import annotations

import hashlib
import torch
from e3nn import o3

from .periodic import gather_neighbors


LINEAR_NAMES = ("readout.direct.weight", "readout.radial_weights.4.weight",
                "readout.radial_weights.4.bias", "learned_scalar_baseline.weight")


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name not in LINEAR_NAMES:
            digest.update(name.encode())
            digest.update(value.detach().cpu().double().numpy().tobytes())
    return digest.hexdigest()


def promote_linear_readout(model):
    """Preserve frozen float32 features; use float64 only after nonlinearities."""
    if not model.uses_maximum_order_readout or model.learned_scalar_baseline is None:
        raise ValueError("Calibration requires maximum-order readout and element baseline")
    model.requires_grad_(False)
    model.readout.direct.double()
    model.readout.tensor_product.double()
    model.readout.radial_weights[-1].double()
    model.learned_scalar_baseline.double()
    return model


def parameter_layout(model, species):
    """Columns of each degree's feature matrix, shared by every radial output."""
    layout = [[] for _ in range(model.config.resolved_density_lmax+1)]
    offset = 0
    for ins in model.readout.direct.instructions:
        size = ins.path_shape[0]*ins.path_shape[1]
        layout[ins.i_out].append(("direct", offset, ins.path_shape[0], ins.path_weight))
        offset += size
    offset = 0
    for index, ins in enumerate(model.readout.tensor_product.instructions):
        u, v, radial = ins.path_shape
        if v != 1 or ins.connection_mode != 'uvw':
            raise ValueError("Unsupported tensor-product parameter layout")
        layout[ins.i_out].append(("tensor",offset,u,index))
        offset += u*radial
    layout[0].append(("baseline", 0, len(species), 0))
    return layout


def linear_parameters(model, species):
    radial = model.config.num_radial
    result = []
    final = model.readout.radial_weights[-1]
    for parts in parameter_layout(model,species):
        blocks = []
        for kind,offset,count,_ in parts:
            if kind == 'direct':
                blocks.append(model.readout.direct.weight[offset:offset+count*radial].reshape(count,radial))
            elif kind == 'tensor':
                w = final.weight[offset:offset+count*radial].reshape(count,radial,-1).permute(0,2,1)
                b = final.bias[offset:offset+count*radial].reshape(count,1,radial)
                blocks.append(torch.cat((w,b),dim=1).reshape(-1,radial))
            else:
                blocks.append(model.learned_scalar_baseline.weight[species])
        result.append(torch.cat(blocks).detach().double())
    return result


@torch.no_grad()
def set_linear_parameters(model, species, parameters):
    radial = model.config.num_radial
    final = model.readout.radial_weights[-1]
    for parts,theta in zip(parameter_layout(model,species),parameters):
        column = 0
        for kind,offset,count,_ in parts:
            width = count*(final.in_features+1) if kind == 'tensor' else count
            values = theta[column:column+width]
            if kind == 'direct':
                model.readout.direct.weight[offset:offset+count*radial].copy_(values.flatten())
            elif kind == 'tensor':
                values = values.reshape(count,final.in_features+1,radial)
                final.weight[offset:offset+count*radial].copy_(values[:,:-1].permute(0,2,1).reshape(-1,final.in_features))
                final.bias[offset:offset+count*radial].copy_(values[:,-1].flatten())
            else:
                model.learned_scalar_baseline.weight[species] = values.to(model.learned_scalar_baseline.weight.dtype)
            column += width
        if column != len(theta):
            raise ValueError("Linear parameter column count differs")


@torch.no_grad()
def frozen_features(model, numbers, positions, mask, cell, species):
    """Return per-degree (atom*m, feature) matrices for one unpadded structure."""
    if positions.shape[0] != 1 or not bool(mask.all()):
        raise ValueError("Use one unpadded structure per feature extraction")
    captured = []
    hook = model.readout.register_forward_pre_hook(lambda module,args: captured.append(args))
    try:
        model(numbers,positions,mask,cell)
    finally:
        hook.remove()
    nodes,harmonics,radial,pair,indices = captured[0]
    source = gather_neighbors(nodes,indices)[0].double()
    h = model.readout.radial_weights[:-1](radial)[0].double()
    h = torch.cat((h,torch.ones_like(h[...,:1])),dim=-1)
    h *= pair[0,...,None]/pair[0].sum(-1).clamp_min(1).double().sqrt()[:,None,None]
    nodes = nodes[0].double()
    harmonics = harmonics[0].double()
    readout = model.readout
    node_slices = readout.tensor_product.irreps_in1.slices()
    edge_slices = readout.tensor_product.irreps_in2.slices()
    matrices = []
    for degree,parts in enumerate(parameter_layout(model,species)):
        width = 2*degree+1
        columns = []
        for kind,_,count,extra in parts:
            if kind == 'direct':
                columns.append((nodes[:,node_slices[degree]].reshape(-1,count,width).transpose(1,2)*extra).reshape(-1,count))
            elif kind == 'tensor':
                ins = readout.tensor_product.instructions[extra]
                l1 = readout.tensor_product.irreps_in1[ins.i_in1].ir.l
                l2 = readout.tensor_product.irreps_in2[ins.i_in2].ir.l
                # Preserve the exact CG constants stored in the trained model.
                cg = getattr(readout.tensor_product._compiled_main_left_right,
                             f'_w3j_{l1}_{l2}_{degree}', None)
                if cg is None:
                    cg = o3.wigner_3j(l1,l2,degree,dtype=torch.float64,device=nodes.device)
                x = source[...,node_slices[ins.i_in1]].reshape(*source.shape[:2],count,2*l1+1)
                y = harmonics[...,edge_slices[ins.i_in2]]
                product = torch.einsum('aeui,aej,ijk->aeuk',x,y,cg)*ins.path_weight
                columns.append(torch.einsum('aeum,aeh->amuh',product,h).reshape(len(nodes)*width,-1))
            else:
                columns.append((numbers[0,:,None] == species[None,:]).double())
        matrices.append(torch.cat(columns,dim=1))
    return matrices


def blocks_to_degree_major(blocks,radial):
    return torch.cat([b.reshape(-1,2*l+1,radial).transpose(1,2).flatten(1)
                      for l,b in enumerate(blocks)],dim=1)
