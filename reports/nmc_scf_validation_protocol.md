# NMC predicted-density SCF validation

## Question

Does the compact coefficient model reduce the number of VASP electronic
iterations relative to VASP's standard superposition-of-atomic-densities
initialization?

Density NMAE alone cannot answer this.  The validation must run paired VASP
calculations with identical structures, POTCARs, k-points, electronic
settings, hardware, and convergence threshold.

## Preflight design

Ten structures from the official NMC test split are used: IDs 1056, 314, 6,
170, 1, 1749, 825, 796, 3, and 361.  They span Li contents from 0 to 12,
mixed transition-metal compositions, and compact-model NMAEs from about
0.89% to 1.90%.  Each structure has five fixed-geometry static calculations:

| Case | Initial density | Purpose |
|---|---|---|
| `sad` | Native VASP `ICHARG=2` | Operational baseline |
| `ml` | ML smooth density plus converged `oracle_full` PAW occupancies | Oracle-assisted compact-model comparison |
| `charge3net` | Official ChargE3Net smooth density plus the same converged PAW occupancies | Paper-style high-accuracy comparison |
| `oracle_smooth` | Exact DFT smooth grid plus the same converged PAW occupancies | Reconstruction/protocol check |
| `oracle_full` | Complete converged DFT CHGCAR | Protocol sanity check; contains data leakage |

The externally supplied smooth-density cases all use the converged
`oracle_full` PAW one-center occupancy matrices.  Consequently, differences
among `ml`, `charge3net`, and `oracle_smooth` isolate the smooth FFT-grid
density.  This follows the protocol reported for ChargE3Net, which does not
predict PAW occupancies.  It is an oracle-assisted comparison rather than an
end-to-end deployable acceleration result.

Common settings are PBEsol (`GGA=PS`), `ALGO=Normal`, `ISPIN=1`, `ISYM=0`,
`PREC=Normal`, a 3x3x1 Monkhorst-Pack grid, `NSW=0`, `EDIFF=1e-6`, and
`LMAXMIX=4`.  `LCHARG` is disabled and every supplied density is copied to
`CHGCAR.initial` before VASP starts, so the model input cannot be overwritten
by the converged output.  These settings follow the NMC dataset where
applicable, but use a tighter static electronic threshold and correct
d-channel restart setting.

## Decision table

| Observation | Interpretation |
|---|---|
| `oracle_full` is not faster than `sad` | Input settings/restart protocol are wrong; do not assess the model |
| `oracle_smooth` and `oracle_full` differ | CHGCAR rewrite or numerical protocol is inconsistent |
| `charge3net` is faster than `ml` | Its more accurate smooth grid improves SCF under identical true PAW data |
| `ml` is faster than `sad` | The compact smooth grid helps in the oracle-PAW-assisted protocol |

The primary endpoint is paired electronic steps to `EDIFF`.  Secondary
endpoints are wall time, VASP elapsed time, failure rate, and final-energy
agreement.  The analyzer also reports end-to-end time after charging each
method for density generation.  This distinction is important because the
compact model takes about 0.5 s per full field while ChargE3Net takes about
578 s on the local GPU.

## Prepared artifacts

The generated benchmark is under `runs/nmc_scf_benchmark_10`.  Set
`VASP_COMMAND` and use `run_all.sh`, or adjust and submit
`submit_slurm.sh`.  After the jobs finish:

```bash
cd /absolute/path/to/nmc_scf_benchmark_10
module load vasp/6.4.0
export VASP_COMMAND="mpirun -np 32 $(command -v vasp_std)"
sbatch --export=ALL,BENCHMARK_ROOT="$PWD" submit_slurm.sh
```

`BENCHMARK_ROOT` is explicit because Slurm executes a copied submission
script from its spool directory on some clusters; deriving the input path
from `$0` is therefore unsafe.

```bash
python scripts/analyze_nmc_scf_benchmark.py \
  --benchmark-dir runs/nmc_scf_benchmark_10
```

The analyzer writes paired JSON and CSV results and reports the mean/median
step reduction and the fraction of structures improved.
