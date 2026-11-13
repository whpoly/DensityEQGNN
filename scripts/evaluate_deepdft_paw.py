"""Evaluate locked PAW heads with a device-safe e3nn Wigner reference.

The installed e3nn keeps Wigner generator constants on CPU. Only the diagnostic
rotation matrix is evaluated on CPU; model outputs, weights, and metrics are
unchanged. Keep the original hashed training/evaluation source intact.
"""
import json
from pathlib import Path
import torch

from scripts import train_deepdft_paw as runner
from scripts.deepdft_paw import IRREPS
from scripts.nmc_probe_holdout import sha256
from scripts.benchmark_deepdft_reuse import save


class DeviceSafeWigner:
    def D_from_matrix(self,rotation):
        return IRREPS.D_from_matrix(rotation.detach().cpu().double()).to(rotation)


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    runner.IRREPS = DeviceSafeWigner()
    save(runner.ROOT/'evaluation_compatibility.json',dict(
        reason='Installed e3nn Wigner generators are CPU constants; initial final run failed in GPU reference-matrix construction.',
        scope='Compute only the diagnostic Wigner reference on CPU float64 then copy to GPU float32. No prediction, training, timing, split, threshold, or checkpoint changes.',
        wrapper_source_sha256=sha256(Path(__file__))))
    runner.final(runner.ROOT)
