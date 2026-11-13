"""Shared radial basis definitions and invertible SVD coordinates.

Whitening retains every radial direction. Evaluate the transformation in
float64: the R16 Gaussian family has a condition number near 5e8.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass(frozen=True)
class RadialSpec:
    num_radial: int = 12
    lmax: int = 4
    cutoff: float = 3.0
    minimum: float = 3.0 / 32.0
    regular: bool = False

    @property
    def size(self):
        return self.num_radial * (self.lmax + 1) ** 2


@lru_cache(maxsize=64)
def radial_transform(spec: RadialSpec, degree: int):
    nodes, weights = np.polynomial.legendre.leggauss(512)
    radius = (nodes + 1) * spec.cutoff / 2
    weights = weights * spec.cutoff / 2 * radius**2
    scales = np.geomspace(spec.minimum, spec.cutoff, spec.num_radial)
    radial = np.exp(-0.5 * (radius[:, None] / scales) ** 2)
    radial *= (0.5 * (1 + np.cos(np.pi * radius / spec.cutoff)))[:, None]
    if spec.regular:
        radial *= (radius[:, None] / scales) ** degree
    _, singular, vh = np.linalg.svd(radial * np.sqrt(weights[:, None]), full_matrices=False)
    if singular[-1] < singular[0] * 1e-12:
        raise ValueError("Radial basis is numerically rank deficient; no implicit truncation allowed")
    return vh.T / singular, singular[:, None] * vh, singular



def radial_values(radius: torch.Tensor, spec: RadialSpec, degree: int, *, whiten: bool,
                  solid_harmonics: bool = False):
    """Evaluate one angular degree, preserving gradients through the radius.

    Autocast is explicitly disabled around the ill-conditioned transform.
    The caller may cast the resulting well-scaled features after this step.
    """
    with torch.autocast(device_type=radius.device.type, enabled=False):
        r = radius.double() if whiten else radius
        scales = torch.as_tensor(
            np.geomspace(spec.minimum, spec.cutoff, spec.num_radial),
            dtype=r.dtype, device=r.device,
        )
        radial = torch.exp(-0.5 * (r[..., None] / scales).square())
        envelope = 0.5 * (1 + torch.cos(torch.pi * r / spec.cutoff))
        radial = radial * (envelope * (r < spec.cutoff))[..., None]
        if spec.regular:
            # Factoring r**l into solid harmonics avoids the normalization
            # epsilon and gives the correct l=1 derivative exactly at r=0.
            radial = radial * ((1 / scales) ** degree if solid_harmonics
                               else (r[..., None] / scales) ** degree)
        if whiten:
            transform = torch.as_tensor(radial_transform(spec, degree)[0],
                                        dtype=torch.float64, device=r.device)
            radial = radial @ transform
        return radial
