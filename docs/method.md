# Method Notes

## Problem

We want a model that can use DFT electron density during training, but does not
require DFT density at inference time.

The training sample is:

```text
{Z_i, R_i, cell}, {r_g, rho_DFT(r_g)}, optional property y
```

The inference input is:

```text
{Z_i, R_i, cell}, requested probe/grid points r_g
```

## Architecture

```text
rho_DFT(r_g), r_g, atoms
  -> DensityToAtomEncoder
  -> z_teacher_i

atoms
  -> ChargeE3LikeAtomNetwork
  -> z_structure_i

z_structure_i
  -> PropertyHead
  -> y_hat

z_structure_i, r_g
  -> AtomToGridDecoder
  -> rho_hat(r_g)
```

The teacher branch makes the atom latent density-informed. The structure branch
learns to reproduce the same latent from atomic geometry and composition alone.

## How this differs from SALTED

SALTED uses a predefined atom-centered density basis:

```text
rho(r) ~= sum_i sum_nlm c_i,nlm R_nl(|r-R_i|) Y_lm(rhat)
```

This project learns the atom-centered latent directly:

```text
z_teacher_i = E_phi(rho_grid, atoms)
```

The decoder then learns how to map the latent back to grid/probe density. This
keeps the VASP/CHGCAR path simple because it does not require SALTED's official
FHI-aims/CP2K/PySCF density-fitting pipeline.

## Relation to ChargE3Net

ChargE3Net predicts density at query/probe points after atom message passing.
This scaffold keeps that atom/probe split and the benchmark choices, while
adding the density-to-atom teacher and optional property supervision.

