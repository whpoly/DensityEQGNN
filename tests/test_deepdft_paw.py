import numpy as np
import torch
from e3nn import o3

from scripts.deepdft_paw import (BLOCKS,IRREPS,WIDTH,PAWReadout,geometry_features,
    potcar_specs,read_structure_and_aug,write_predicted_aug,training_statistics)


def fixture():
    torch.manual_seed(71)
    n = 5
    xyz = torch.randn(n,3)*.4
    edges = torch.tensor([(i,j) for i in range(n) for j in range(n) if i!=j])
    batch = dict(atom_xyz=xyz[None],atom_edges=edges[None],
        atom_edges_displacement=torch.zeros(1,len(edges),3),cell=torch.eye(3)[None]*9)
    return batch,torch.randn(n,384),torch.randn(n,3,384),torch.tensor([3,8,25,27,28])


def test_head_rotation_inversion_translation_and_gradients():
    batch,s,v,z = fixture()
    head = PAWReadout(torch.zeros(29,WIDTH),torch.ones(29,WIDTH))
    # Nonzero output is essential: zero-init would make equivariance vacuous.
    torch.nn.init.normal_(head.edge[-1].weight,std=.05)
    torch.nn.init.normal_(head.onsite.weight,std=.05)
    q = torch.eye(3)[[1,2,0]]
    for rotation in (o3.rand_matrix(),-o3.rand_matrix()):
        transformed = dict(batch,atom_xyz=batch['atom_xyz']@rotation.T+torch.tensor([.2,-.3,.4]))
        out = head(s,v,z,geometry_features(batch))
        changed = head(s,torch.einsum('ij,ajc->aic',rotation,v),z,geometry_features(transformed))
        expected = out@IRREPS.D_from_matrix(q@rotation@q.T).T
        torch.testing.assert_close(changed,expected,rtol=2e-4,atol=1e-5)
    out.square().mean().backward()
    for name,p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert head.vector.weight.grad.abs().sum()>0


def test_native_paw_roundtrip_and_mask(tmp_path):
    specs = potcar_specs()
    sample = read_structure_and_aug('data/nmc_cache/0.CHGCAR.lz4',specs,lmax=2)
    path = tmp_path/'CHGCAR'
    write_predicted_aug(sample,sample['target_aug'].numpy(),path)
    reread = read_structure_and_aug(path,specs,lmax=2)
    torch.testing.assert_close(sample['target_aug'],reread['target_aug'],rtol=0,atol=0)
    assert sample['_source_text'].split('augmentation')[0]==path.read_text().split('augmentation')[0]
    for raw,mapping,row in zip(sample['raw_aug'],sample['raw_mapping'],sample['target_aug']):
        np.testing.assert_allclose(raw,row.numpy()[mapping],rtol=1e-7,atol=1e-7)
    for b in BLOCKS:
        if b['l']>2:
            assert not sample['aug_mask'][:,b['start']:b['stop']].any()
    mean,scale = training_statistics([sample])
    for b in BLOCKS:
        if b['l']>0:
            assert torch.count_nonzero(mean[:,b['start']:b['stop']])==0
        torch.testing.assert_close(scale[:,b['start']:b['stop']],scale[:,b['start'],None].expand(-1,2*b['l']+1))
