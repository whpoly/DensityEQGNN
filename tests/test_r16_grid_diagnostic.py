import torch
from density_eqgnn.periodic import cutoff_edges
from scripts.r16_grid_diagnostic import uniform_grid_edges


def canonical(edges,probes,positions,cell):
    # Discrete image identity distinguishes multiple copies of the same atom.
    # Neighbor-minus-center displacement gives the image from d + q - p.
    shifts=torch.round((edges.displacement+probes[edges.center]-positions[edges.neighbor])@torch.linalg.inv(cell)).long()
    result={}
    for i,(c,n,s) in enumerate(zip(edges.center.tolist(),edges.neighbor.tolist(),shifts.tolist())):
        key=(c,n,*s)
        assert key not in result
        result[key]=edges.displacement[i]
    return result


def test_local_grid_stencil_matches_all_images_in_skew_cell_with_unwrapped_atoms():
    torch.manual_seed(84)
    cell=torch.tensor([[2.6,0.,0.],[1.1,2.9,0.],[.5,.7,3.1]],dtype=torch.float64)
    shape=(7,8,9)
    frac=torch.stack(torch.meshgrid(*[torch.arange(n,dtype=torch.float64)/n for n in shape],indexing='ij'),-1).reshape(-1,3)
    probes=frac@cell
    positions=torch.tensor([[1.1,-.2,.45],[.32,.64,.11]],dtype=torch.float64)@cell
    old=cutoff_edges(probes[None],positions[None],torch.ones(1,len(probes),dtype=torch.bool),
                     torch.ones(1,2,dtype=torch.bool),cell[None],3.,True)
    new=uniform_grid_edges(positions,cell,shape,3.,atom_chunk=1)
    a,b=canonical(old,probes,positions,cell),canonical(new,probes,positions,cell)
    assert a.keys()==b.keys()
    assert len(a)>len(probes)*len(positions) # Includes multiple periodic images.
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=1e-12,atol=1e-12)
