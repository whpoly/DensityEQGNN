"""Verify the reusable label-free API against the previous full-grid path."""
from pathlib import Path
import torch
from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_edengnn import atom_data,make_sad,write_json
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_decoder_diagnostic import decode
from scripts.fit_nmc_holdout_oracle import BASIS


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    predictor=FastAdaptiveR16()
    sample=load_sample(Path('data/nmc_cache/0.CHGCAR.lz4'),'cuda')
    actual=predictor(sample['numbers'],sample['positions'],sample['cell'],sample['metadata']['grid_shape']).flatten()
    data=atom_data(sample)
    coeff=predictor.model.coefficients(data)
    edges=uniform_grid_edges(sample['positions'],sample['cell'],sample['metadata']['grid_shape'],BASIS.cutoff)
    expected=decode(coeff,-edges.displacement,edges.batch,edges.neighbor,edges.center,len(actual),BASIS).double()
    expected+=make_sad(sample,read_form_factors(Path('potpaw54.zip'))).double()
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=5e-5)
    audit={'sample_id':0,'grid_points':len(actual),'max_abs_difference':float((actual-expected).abs().max()),
           'relative_l1_difference':float((actual-expected).abs().sum()/expected.abs().sum()),
           'api_arguments':'numbers, positions, cell, grid_shape; no target density'}
    write_json('runs/nmc_inference_fast_2026-10-29/api_audit.json',audit)
    print(audit)


if __name__=='__main__':
    main()
