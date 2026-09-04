"""Utilities for constructing Gaussian mixture approximations.

This module implements the mixture construction strategy from:
    Chen, Villa, Ghattas (2024)
    "Gaussian mixture Taylor approximations for risk measures of PDEs"

The key idea is to approximate N(m_bar, C) by a Gaussian mixture with
reduced variance along a dominant direction (KLE or HEP eigenvector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import dolfin as dl
from hippylib import MultiVector, Random
from hippylib.algorithms.randomizedEigensolver import doublePassG

from .mixture_data import get_1d_mixture


@dataclass
class MixtureComponent:
    """A single Gaussian mixture component.

    Attributes:
        weight: Mixture weight w_i
        mean: Component mean m_bar_i (dolfin Vector)
        sigma_scale: Variance scaling factor sigma_i^2
        lambda_psi: Pseudo-eigenvalue for the decomposition direction
        psi: Decomposition direction (dolfin Vector, C^{-1}-orthonormalized)
        C_inv_psi: C^{-1} @ psi for efficient computations
    """
    weight: float
    mean: dl.Vector
    sigma_scale: float  # sigma_i^2
    lambda_psi: float
    psi: dl.Vector
    C_inv_psi: dl.Vector


def compute_kle_direction(prior, n_modes=1):
    """Compute dominant KLE eigenvector(s) of the covariance operator.

    Solves: C @ phi = lambda @ phi

    Args:
        prior: HippyLib prior with covariance operator
        n_modes: Number of dominant modes to compute

    Returns:
        Tuple of (eigenvalues, eigenvectors as MultiVector)
    """
    # Use randomized eigendecomposition
    # For KLE, we solve C @ phi = lambda @ phi
    # This is equivalent to R^{-1} @ phi = lambda @ phi where R = C^{-1}

    Vh = prior.Vh
    omega = MultiVector(dl.Function(Vh).vector(), n_modes + 10)
    rand = Random()
    for i in range(n_modes + 10):
        rand.normal(1.0, omega[i])

    # Solve generalized eigenvalue problem: C @ phi = lambda @ I @ phi
    # Using doublePassG with A=C (via Rsolver), B=I (identity), we get KLE modes
    # Actually, we need to be careful here. The standard setup uses:
    # A @ v = lambda @ B @ v where A = Hessian, B = R (precision)
    #
    # For KLE, we want C @ phi = lambda @ phi
    # This is equivalent to: I @ phi = lambda @ R @ phi (multiply by R=C^{-1})
    # Or in doublePassG terms with A=I, B=R: I @ phi = lambda @ R @ phi

    # Simpler approach: use prior's methods if available
    # Or use power iteration on C

    # For now, use a simple approach with prior sampling
    d = np.zeros(n_modes)
    U = MultiVector(dl.Function(Vh).vector(), n_modes)

    # Power iteration to get dominant eigenvector
    v = dl.Function(Vh).vector()
    rand.normal(1.0, v)

    # Apply C multiple times (power iteration)
    for _ in range(50):
        # w = C @ v
        w = dl.Function(Vh).vector()
        prior.Rsolver.solve(w, v)  # w = C @ v (since Rsolver applies C = R^{-1})

        # Normalize
        norm_w = np.sqrt(w.inner(w))
        if norm_w > 1e-14:
            v.zero()
            v.axpy(1.0 / norm_w, w)

    # The dominant eigenvalue
    w = dl.Function(Vh).vector()
    prior.Rsolver.solve(w, v)  # w = C @ v
    d[0] = v.inner(w) / v.inner(v)
    U[0].zero()
    U[0].axpy(1.0, v)

    return d, U


def compute_hep_direction(H, prior, n_modes=1):
    """Compute dominant Hessian eigenvalue problem (HEP) eigenvector(s).

    Solves the generalized eigenvalue problem:
        D^2 Q(m_bar) @ phi = lambda @ C^{-1} @ phi

    This direction balances variance (from C) and nonlinearity (from Hessian).

    Args:
        H: ReducedHessianSVD object (already set at linearization point)
        prior: HippyLib prior with precision operator R = C^{-1}
        n_modes: Number of dominant modes to compute

    Returns:
        Tuple of (eigenvalues, eigenvectors as MultiVector)
    """
    Vh = H.pde.Vh[1]  # Parameter space
    omega = MultiVector(dl.Function(Vh).vector(), n_modes + 10)
    rand = Random()
    for i in range(n_modes + 10):
        rand.normal(1.0, omega[i])

    # Solve: H @ phi = lambda @ R @ phi using doublePassG
    d, U = doublePassG(H, prior.R, prior.Rsolver, omega, n_modes, s=1)

    return d, U


def construct_mixture_components(
    prior,
    psi: dl.Vector,
    lambda_psi: float,
    n_components: int,
) -> List[MixtureComponent]:
    """Construct Gaussian mixture components from 1D mixture and direction.

    Given decomposition direction psi and 1D mixture approximation of N(0,1),
    constructs the full Gaussian mixture approximation of N(m_bar, C).

    The component means and covariances are:
        m_bar_i = m_bar + mu_i * sqrt(lambda_psi) * psi
        C_i = C + (sigma_i^2 - 1) * lambda_psi * psi @ psi^T

    Args:
        prior: HippyLib prior with mean and covariance
        psi: Decomposition direction (should be C^{-1}-orthonormal: <psi, C^{-1} psi> = 1)
        lambda_psi: Pseudo-eigenvalue = <psi, C^{-1} psi>^{-1}
        n_components: Number of mixture components

    Returns:
        List of MixtureComponent objects
    """
    # Get 1D mixture approximation
    mix_1d = get_1d_mixture(n_components)
    weights = mix_1d['weights']
    means_1d = mix_1d['means']
    sigma = mix_1d['sigma']
    sigma_sq = sigma ** 2

    # Pre-compute C^{-1} @ psi for efficiency
    C_inv_psi = dl.Function(prior.Vh).vector()
    prior.R.mult(psi, C_inv_psi)  # C_inv_psi = R @ psi = C^{-1} @ psi

    components = []
    sqrt_lambda_psi = np.sqrt(lambda_psi)

    for i in range(n_components):
        # Component mean: m_bar_i = m_bar + mu_i * sqrt(lambda_psi) * psi
        mean_i = dl.Function(prior.Vh).vector()
        mean_i.axpy(1.0, prior.mean)
        mean_i.axpy(means_1d[i] * sqrt_lambda_psi, psi)

        # Store component
        comp = MixtureComponent(
            weight=weights[i],
            mean=mean_i,
            sigma_scale=sigma_sq,
            lambda_psi=lambda_psi,
            psi=psi,
            C_inv_psi=C_inv_psi,
        )
        components.append(comp)

    return components


def compute_component_linear_variance(g, prior, component: MixtureComponent):
    """Compute variance of linear Taylor approximation for a mixture component.

    For component i with covariance C_i = C + (sigma^2 - 1) * lambda_psi * psi @ psi^T,
    the linear variance is:
        <g, C_i @ g> = <g, C @ g> + (sigma^2 - 1) * lambda_psi * <g, psi>^2

    Args:
        g: Gradient vector DQ(m_bar_i)
        prior: HippyLib prior
        component: MixtureComponent object

    Returns:
        Linear variance <g, C_i @ g>
    """
    # Compute <g, C @ g> using prior
    Cg = dl.Function(prior.Vh).vector()
    prior.Rsolver.solve(Cg, g)  # Cg = C @ g = R^{-1} @ g
    var_C = g.inner(Cg)

    # Compute correction term: (sigma^2 - 1) * lambda_psi * <g, psi>^2
    g_dot_psi = g.inner(component.psi)
    correction = (component.sigma_scale - 1.0) * component.lambda_psi * g_dot_psi ** 2

    return var_C + correction


def compute_pseudo_eigenvalue(psi, prior):
    """Compute pseudo-eigenvalue lambda_psi = <psi, C^{-1} psi>^{-1}.

    Args:
        psi: Direction vector
        prior: HippyLib prior with precision operator R = C^{-1}

    Returns:
        Pseudo-eigenvalue lambda_psi
    """
    # Compute <psi, R @ psi> = <psi, C^{-1} @ psi>
    R_psi = dl.Function(prior.Vh).vector()
    prior.R.mult(psi, R_psi)
    inner_prod = psi.inner(R_psi)

    if inner_prod < 1e-14:
        raise ValueError("Direction psi has near-zero C^{-1}-norm")

    return 1.0 / inner_prod


def normalize_direction(psi, prior):
    """Normalize direction to have unit C^{-1}-norm.

    Makes psi satisfy <psi, C^{-1} psi> = 1, so lambda_psi = 1.

    Args:
        psi: Direction vector (modified in place)
        prior: HippyLib prior

    Returns:
        Normalization factor
    """
    # Compute <psi, R @ psi>
    R_psi = dl.Function(prior.Vh).vector()
    prior.R.mult(psi, R_psi)
    norm_sq = psi.inner(R_psi)

    if norm_sq < 1e-14:
        raise ValueError("Direction psi has near-zero C^{-1}-norm")

    norm = np.sqrt(norm_sq)
    psi.apply("")  # Ensure vector is assembled
    psi_local = psi.get_local()
    psi.set_local(psi_local / norm)
    psi.apply("")

    return norm


__all__ = [
    'MixtureComponent',
    'compute_kle_direction',
    'compute_hep_direction',
    'construct_mixture_components',
    'compute_component_linear_variance',
    'compute_pseudo_eigenvalue',
    'normalize_direction',
]
