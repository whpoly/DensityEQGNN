import torch

from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, atom_batch
from scripts.deepdft_inference_fast import ProbeEdges
from scripts.deepdft_mixed_embedding import MixedProbe, TrainingEdges, descriptor
from test_deepdft_inference_fast import sample


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    backbone, _ = official_model(DEFAULT_ROOT/'DeepDFT', 'cpu')
    s = sample()
    with torch.no_grad():
        rep = backbone.atom_model(atom_batch(s))
    return s, rep


def test_compact_edges_duplicates_and_unordered_queries():
    s, _ = setup()
    graph = ProbeEdges.build(s)
    compact = TrainingEdges(graph)
    ids = torch.tensor([41, 0, 41, 119, 8])
    source, receiver, diff = compact.select(s, ids)
    for k, idx in enumerate(ids.tolist()):
        a, _, d = graph.chunk(s, idx, idx+1)
        torch.testing.assert_close(source[receiver==k], a)
        torch.testing.assert_close(diff[receiver==k], d)


def test_global_invariance_conditioning_and_constant_control():
    s, rep = setup()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = (rep[0], [torch.einsum('ij,ajc->aic', q, v) for v in rep[1]])
    torch.testing.assert_close(descriptor(rep), descriptor(rotated), rtol=1e-5, atol=1e-6)
    perm = torch.arange(rep[0][0].shape[0]-1, -1, -1)
    shuffled = ([x[perm] for x in rep[0]], [x[perm] for x in rep[1]])
    torch.testing.assert_close(descriptor(rep), descriptor(shuffled))
    head = MixedProbe('global', 16)
    torch.nn.init.normal_(head.global_encoder[-1].weight, std=.03)
    graph = ProbeEdges.build(s)
    a, b, d = graph.chunk(s, 0, 35)
    output = head(a, b, d, *head.atoms(rep), 35)
    turned = head(a, b, d @ q.T, *head.atoms(rotated), 35)
    torch.testing.assert_close(output, turned, rtol=1e-4, atol=1e-5)
    changed = head(a, b, d, *head.atoms(rep, descriptor(rep)+1), 35)
    assert (output-changed).abs().max() > 1e-5
    control = MixedProbe('constant', 16)
    control.load_state_dict(head.state_dict())
    x = control.atoms(rep)[-1]
    y = control.atoms(rep, descriptor(rep)+1)[-1]
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    output.sum().backward()
    assert head.global_encoder[-1].weight.grad.abs().sum() > 0


def test_initial_pairing_and_chunk_independence():
    s, rep = setup()
    heads = {m:MixedProbe(m, 16) for m in ('local', 'constant', 'global')}
    for h in heads.values():
        h.load_state_dict(heads['global'].state_dict())
    graph = ProbeEdges.build(s)
    outputs = []
    for h in heads.values():
        states = h.atoms(rep)
        full = h(*graph.chunk(s, 0, graph.size), *states, graph.size)
        parts = [h(*graph.chunk(s, i, min(i+17, graph.size)), *states, min(17, graph.size-i))
                 for i in range(0, graph.size, 17)]
        torch.testing.assert_close(full, torch.cat(parts), atol=1e-6, rtol=1e-5)
        outputs.append(full)
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], atol=0, rtol=0)
