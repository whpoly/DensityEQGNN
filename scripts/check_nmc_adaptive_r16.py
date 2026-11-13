"""Real-data warm-start, decoder and graph-gradient preflight; no fitting."""
import torch
from scripts.nmc_adaptive_r16 import ROOT,build
from scripts.nmc_eden_r16 import prepare,evaluate
from scripts.nmc_edengnn import write_json
from scripts.train_nmc_eden_r16 import decoder_audit,graph_audit


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    model=build('adaptive').eval()
    baseline=build('continued').eval()
    samples=prepare(model,'eden_r16_sad')
    warm=[]
    with torch.no_grad():
        for s in samples:
            encoded=s['atom_data'].clone()
            model.encoder(encoded)
            delta=model.adaptive_degree(encoded['node_features'],encoded.z)
            assert torch.count_nonzero(delta)==0
            a=model.coefficients(s['atom_data'])
            b=baseline.coefficients(s['atom_data'])
            # Independent CUDA scatter reductions are not bitwise deterministic.
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            warm.append({'sample_id':s['id'],'max_abs_coefficient_difference':float((a-b).abs().max()),
                         'adaptive_residual_exact_zero':True})
    del baseline
    result={'warm_start':warm,'decoder':decoder_audit(model,samples,'eden_r16_sad')}
    model.capture(samples)
    result['graph']=graph_audit(model,samples)
    result['validation']=evaluate(model,samples,'eden_r16_sad')
    assert abs(result['validation']['selection_macro_nmae']-0.0034417719262500855)<2e-8
    result['parameters']=sum(p.numel() for p in model.parameters())
    write_json(ROOT/'preflight.json',result)
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':
    main()
