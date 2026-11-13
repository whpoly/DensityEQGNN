import torch

from scripts.deepdft_paw import PAWReadout,WIDTH,geometry_features
from scripts.public_paw_experiment import collate,predict,structure_loss


def test_disjoint_batch_preserves_predictions_and_structure_weighted_gradients():
    torch.manual_seed(71)
    rows=[]
    for n in (3,5):
        xyz=torch.randn(n,3)*.4
        edges=torch.tensor([(i,j) for i in range(n) for j in range(n) if i!=j])
        batch=dict(atom_xyz=xyz[None],atom_edges=edges[None],
            atom_edges_displacement=torch.zeros(1,len(edges),3),cell=torch.eye(3)[None]*9)
        z=torch.tensor([3,8,25,27,28])[:n]
        rows.append(dict(numbers=z,scalars=torch.randn(n,384),vectors=torch.randn(n,3,384),
            geometry=geometry_features(batch),target_aug=torch.randn(len(z),WIDTH),
            aug_mask=torch.rand(len(z),WIDTH)>.3))
    head=PAWReadout(torch.zeros(29,WIDTH),torch.ones(29,WIDTH),lmax=2)
    torch.nn.init.normal_(head.edge[-1].weight,std=.02)
    b=collate(rows,'deep_frozen')
    out=predict(head,b,'deep_frozen')
    independent=[predict(head,collate([r],'deep_frozen'),'deep_frozen') for r in rows]
    torch.testing.assert_close(out,torch.cat(independent),rtol=2e-5,atol=1e-6)
    loss=structure_loss(out,b,head.scale)
    expected=torch.stack([((p-r['target_aug'])**2)[r['aug_mask']].mean() for p,r in zip(independent,rows)]).mean()
    torch.testing.assert_close(loss,expected)
    gradients=torch.autograd.grad(loss,tuple(head.parameters()),retain_graph=True)
    reference=torch.autograd.grad(expected,tuple(head.parameters()))
    for a,b in zip(gradients,reference): torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
