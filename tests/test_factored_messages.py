import torch

from scripts.deepdft_factored_messages import FactoredProbe
from scripts.deepdft_cubic_far import expanded_far_density
from scripts.deepdft_inference_fast import ProbeEdges,flat_density
from scripts.nmc_probe_holdout import atom_batch
from test_deepdft_near_far import fixture


@torch.no_grad()
def test_factored_aggregates_match_original_layer_states():
    model,core,sample = fixture()
    fast = FactoredProbe(model).eval()
    rep = model.atom_model(atom_batch(sample))
    original,prepared = core.atoms(rep),fast.atoms(rep)
    source,receiver,diff = ProbeEdges.build(sample).chunk(sample,0,len(sample['probes']))
    geometry = core.geometry(diff)
    for k,layer in enumerate(core.probe.message_layers):
        a = core.aggregate(layer,*(s[k] for s in original),source,receiver,geometry,len(sample['probes']))
        b = fast.aggregate(layer,*(s[k] for s in prepared),source,receiver,geometry,len(sample['probes']))
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-6)


@torch.no_grad()
def test_factored_full_and_cubic_density_refresh_and_chunks():
    model,core,sample = fixture()
    fast = FactoredProbe(model).eval()
    changed = dict(sample,positions=sample['positions']+.03,numbers=torch.tensor([28,8]))
    for s in (sample,changed):
        torch.testing.assert_close(flat_density(model,fast,s,chunk=17),flat_density(model,core,s,chunk=31),rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(expanded_far_density(model,fast,s,radius=3.,chunk=17),
            expanded_far_density(model,core,s,radius=3.,chunk=31),rtol=3e-5,atol=3e-6)
