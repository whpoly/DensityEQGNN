"""Post-run numerical verification; never selects or modifies any model."""
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save
from scripts.benchmark_joint_density import ROOT,check


def errors(a,b):
    a,b = a.astype(np.float64),b.astype(np.float64)
    return dict(max_abs=float(np.abs(a-b).max()),relative_l1=float(np.abs(a-b).sum()/np.abs(b).sum()))


def main():
    cfg = check(ROOT)
    records = []
    for name in cfg['variants']:
        final = json.loads((ROOT/'final'/f'{name}.json').read_text())
        assert final['completed'] and final['source_hashes_match'] and final['checkpoint_hashes_match']
        for row in final['records']:
            if 'error' in row or name=='eden':
                continue
            sid = row['id']
            ref = np.load(ROOT/'predictions'/'deep_reference'/f'{sid}.npz')
            new = np.load(ROOT/'predictions'/name/f'{sid}.npz')
            record = dict(variant=name,id=sid,density=errors(new['density'],ref['density']),paw={})
            for seed in (42,43):
                a,b = new[f'augmentation{seed}'],ref[f'augmentation{seed}']
                record['paw'][str(seed)] = errors(a,b)
                np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5)
                assert record['paw'][str(seed)]['relative_l1']<1e-6
            if name.startswith('fused_cubic'):
                shape = row['metadata']['grid_shape']
                a = new['density'].reshape(shape)[::2,::2,::2]
                b = ref['density'].reshape(shape)[::2,::2,::2]
                record['coarse_anchors'] = errors(a,b)
                np.testing.assert_allclose(a,b,rtol=5e-5,atol=5e-5)
                assert record['coarse_anchors']['relative_l1']<1e-6
            else:
                np.testing.assert_allclose(new['density'],ref['density'],rtol=5e-5,atol=5e-5)
                assert record['density']['relative_l1']<1e-6
            records.append(record)
            ref.close(); new.close()
    # Check copied DFT sources themselves, not just cached tensors.
    from scripts.nmc_probe_holdout import sha256
    from scripts.joint_density_predictor import DEEP_PAW,EDEN_PAW
    sources_unchanged = all(sha256(r['source'])==r['source_sha256'] for r in cfg['records'])
    assert sources_unchanged
    previous = json.loads((EDEN_PAW/'settings.json').read_text())
    assert sha256(DEEP_PAW/'dataset.pt')==cfg['base_paw_dataset_sha256']==previous['base_dataset_sha256']
    save(ROOT/'output_verification.json',dict(passed=True,records=records,original_data_unchanged=True,
        original_checkpoints_unchanged=True,source_hashes_match=True,verifier_sha256=sha256(__file__)))
    print(json.dumps(dict(passed=True,comparisons=len(records),max_paw_relative_l1=max(
        v['relative_l1'] for r in records for v in r['paw'].values()))))


if __name__=='__main__':
    main()
