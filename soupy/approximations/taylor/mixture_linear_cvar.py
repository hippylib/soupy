"""Gaussian mixture with linear Taylor approximation for CVaR.

This implements the Gaussian mixture Taylor approximation for CVaR from:
    Chen, Villa, Ghattas (2024)
    "Gaussian mixture Taylor approximations for risk measures of PDEs"

The key idea is to approximate the prior N(m_bar, C) by a Gaussian mixture
with reduced variance along a dominant direction, then use linear Taylor
approximations at each mixture component mean.

For the linear case, each component Q_lin,i is Gaussian, so we can compute
the CVaR of the Gaussian mixture analytically.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import dolfin as dl
from hippylib import Random, vector2Function, MultiVector
from hippylib.algorithms.randomizedEigensolver import doublePassG
from scipy.stats import norm
from scipy.optimize import brentq

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL
from .settings import taylor_mixture_linear_cvar_settings
from .mixture_data import get_1d_mixture
from .linear_cvar import gaussian_cvar


def gaussian_mixture_cvar(means, stds, weights, beta, tol=1e-10):
    """Compute CVaR of a Gaussian mixture distribution.

    For Q_mix ~ sum_i w_i N(mu_i, sigma_i^2), computes:
        CVaR_beta[Q_mix] = min_t { t + E[(Q_mix - t)^+] / (1-beta) }

    The VaR (quantile) is found by solving:
        sum_i w_i * P[Q_i > t] = 1 - beta

    Then CVaR is computed using the closed-form expression for
    E[(X - t)^+] where X ~ N(mu, sigma^2).

    Args:
        means: Array of component means
        stds: Array of component standard deviations
        weights: Array of mixture weights (should sum to 1)
        beta: CVaR risk level (e.g., 0.95)
        tol: Tolerance for root finding

    Returns:
        Tuple of (var, cvar) - Value-at-Risk and Conditional Value-at-Risk
    """
    means = np.asarray(means)
    stds = np.asarray(stds)
    weights = np.asarray(weights)

    n_components = len(means)

    # Special case: single component (standard Gaussian CVaR)
    if n_components == 1:
        cvar = gaussian_cvar(means[0], stds[0], beta)
        var = means[0] + stds[0] * norm.ppf(beta)
        return var, cvar

    # Find VaR by solving: sum_i w_i * P[Q_i > t] = 1 - beta
    # This is equivalent to: sum_i w_i * (1 - Phi((t - mu_i)/sigma_i)) = 1 - beta

    def exceedance_prob(t):
        """Probability that Q_mix > t."""
        prob = 0.0
        for i in range(n_components):
            if stds[i] > 1e-14:
                prob += weights[i] * (1.0 - norm.cdf((t - means[i]) / stds[i]))
            else:
                # Degenerate case: point mass
                prob += weights[i] * (1.0 if means[i] > t else 0.0)
        return prob

    def var_equation(t):
        """Equation to solve for VaR: P[Q > t] - (1-beta) = 0."""
        return exceedance_prob(t) - (1.0 - beta)

    # Find bounds for root finding
    mean_min = np.min(means - 4 * stds)
    mean_max = np.max(means + 4 * stds)

    # Ensure we have a sign change
    f_min = var_equation(mean_min)
    f_max = var_equation(mean_max)

    if f_min * f_max > 0:
        # No sign change - try expanding bounds
        mean_min = np.min(means - 10 * stds)
        mean_max = np.max(means + 10 * stds)
        f_min = var_equation(mean_min)
        f_max = var_equation(mean_max)

    if f_min * f_max > 0:
        # Still no sign change - return approximate value
        var = np.sum(weights * (means + stds * norm.ppf(beta)))
        cvar = np.sum(weights * np.array([gaussian_cvar(m, s, beta) for m, s in zip(means, stds)]))
        return var, cvar

    # Find VaR using Brent's method
    try:
        var = brentq(var_equation, mean_min, mean_max, xtol=tol)
    except ValueError:
        # Fallback to weighted average
        var = np.sum(weights * (means + stds * norm.ppf(beta)))

    # Compute CVaR using formula for E[(X - t)^+] for Gaussian X
    # E[(X - t)^+] = sigma * phi((t-mu)/sigma) + (mu - t) * (1 - Phi((t-mu)/sigma))
    cvar = var
    for i in range(n_components):
        mu_i, sigma_i, w_i = means[i], stds[i], weights[i]
        if sigma_i > 1e-14:
            z = (var - mu_i) / sigma_i
            # E[(Q_i - var)^+]
            expected_excess = sigma_i * norm.pdf(z) + (mu_i - var) * (1.0 - norm.cdf(z))
        else:
            # Degenerate case
            expected_excess = max(0.0, mu_i - var)
        cvar += w_i * expected_excess / (1.0 - beta)

    return var, cvar


class _TaylorMixtureLinearCVaRLegacy:
    """Implementation of Gaussian mixture linear Taylor CVaR."""

    def __init__(self, settings, model, prior, penalization, tol=1e-9):
        self.settings = settings
        self.model = model
        self.pde = model.problem
        self.qoi = model.qoi
        self.prior = prior
        self.penalization = penalization
        self.tol = tol

        self.z = model.generate_vector(CONTROL)
        self.m = prior.mean
        self.x = model.generate_vector(STATE)
        self.y = model.generate_vector(STATE)
        self.x_all = [self.x, self.m, self.y, self.z]

        self.z_dir = model.generate_vector(CONTROL)
        self.grad_cache = None

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.beta = settings["beta"]
        self.N_mix = settings["N_mix"]
        self.direction = settings["direction"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        # MPI setup
        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        # Get 1D mixture parameters
        self.mix_1d = get_1d_mixture(self.N_mix)

        # Storage for mixture component quantities
        self.component_means = []  # Q(m_bar_i)
        self.component_stds = []  # sqrt(<g_i, C_i @ g_i>)
        self.component_weights = self.mix_1d['weights']

        # Direction and components (computed on first objective call)
        self.psi = None  # Decomposition direction
        self.lambda_psi = None  # Pseudo-eigenvalue
        self.m_bar_i = []  # Component means in parameter space
        self.direction_computed = False

        # Hessian for HEP direction
        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        # CVaR values
        self.var = 0.0
        self.cvar = 0.0

        # For gradient computation
        self.dominant_component_idx = 0

    def _compute_direction(self):
        """Compute decomposition direction (KLE or HEP)."""
        if self.direction_computed:
            return

        # Solve forward and adjoint at prior mean first
        self.x_all[CONTROL] = self.z
        self.x_all[PARAMETER] = self.prior.mean
        self.pde.solveFwd(self.x, self.x_all)
        self.x_all[STATE] = self.x

        rhs = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(self.x_all, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs)
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False)
        self.qoi.setLinearizationPoint(self.x_all)

        if self.direction == "hep":
            # Compute dominant HEP eigenvector
            omega = MultiVector(self.pde.generate_parameter(), 15)
            rand = Random()
            for i in range(15):
                rand.normal(1.0, omega[i])

            d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, 1, s=1)

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.axpy(1.0, U[0])
            self.lambda_psi = 1.0  # U is already C^{-1}-orthonormal

            if self.verbose and self.mpi_rank == 0:
                print(f"  [Mixture] Using HEP direction, dominant eigenvalue = {d[0]:.4e}")

        else:  # KLE direction
            # Use power iteration to get dominant KLE eigenvector
            v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            rand = Random()
            rand.normal(1.0, v)

            for _ in range(50):
                w = dl.Function(self.pde.Vh[PARAMETER]).vector()
                self.prior.Rsolver.solve(w, v)  # w = C @ v
                norm_w = np.sqrt(w.inner(w))
                if norm_w > 1e-14:
                    v.zero()
                    v.axpy(1.0 / norm_w, w)

            # Normalize to have unit C^{-1}-norm
            R_v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.prior.R.mult(v, R_v)
            norm_sq = v.inner(R_v)
            v_local = v.get_local()
            v.set_local(v_local / np.sqrt(norm_sq))
            v.apply("")

            self.psi = v
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print(f"  [Mixture] Using KLE direction")

        # Construct mixture component means in parameter space
        mix_means_1d = self.mix_1d['means']
        sqrt_lambda = np.sqrt(self.lambda_psi)

        self.m_bar_i = []
        for i in range(self.N_mix):
            m_i = dl.Function(self.pde.Vh[PARAMETER]).vector()
            m_i.axpy(1.0, self.prior.mean)
            m_i.axpy(mix_means_1d[i] * sqrt_lambda, self.psi)
            self.m_bar_i.append(m_i)

        self.direction_computed = True

    def objective(self):
        """Compute CVaR objective using Gaussian mixture linear Taylor."""
        self._compute_direction()

        sigma_1d = self.mix_1d['sigma']
        sigma_sq = sigma_1d ** 2

        self.component_means = []
        self.component_stds = []

        # Pre-compute R @ psi = C^{-1} @ psi
        R_psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
        self.prior.R.mult(self.psi, R_psi)

        for i in range(self.N_mix):
            m_i = self.m_bar_i[i]

            # Solve forward at component mean m_i
            x_i = self.model.generate_vector(STATE)
            y_i = self.model.generate_vector(STATE)
            x_all_i = [x_i, m_i, y_i, self.z]

            self.pde.solveFwd(x_i, x_all_i)
            x_all_i[STATE] = x_i

            # Compute QoI at component mean
            Q_i = self.qoi.cost(x_all_i)
            self.component_means.append(Q_i)

            # Solve adjoint at component mean
            rhs_i = self.model.generate_vector(STATE)
            self.qoi.adj_rhs(x_all_i, rhs_i)
            self.pde.solveAdj(y_i, x_all_i, rhs_i)
            x_all_i[ADJOINT] = y_i

            # Compute gradient g_i = DQ(m_i) w.r.t. parameter
            self.pde.setLinearizationPoint(x_all_i, False)
            self.qoi.setLinearizationPoint(x_all_i)

            x_fun = vector2Function(x_i, self.pde.Vh[STATE])
            y_fun = vector2Function(y_i, self.pde.Vh[ADJOINT])
            m_fun = vector2Function(m_i, self.pde.Vh[PARAMETER])
            z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])

            form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
            m_test = dl.TestFunction(self.pde.Vh[PARAMETER])
            g_i = dl.assemble(dl.derivative(form, m_fun, m_test))

            # Compute variance <g_i, C_i @ g_i>
            # C_i = C + (sigma^2 - 1) * lambda_psi * psi @ psi^T
            # So: <g_i, C_i @ g_i> = <g_i, C @ g_i> + (sigma^2 - 1) * lambda_psi * <g_i, psi>^2

            # <g_i, C @ g_i>
            C_g_i = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.prior.Rsolver.solve(C_g_i, g_i)
            var_C = g_i.inner(C_g_i)

            # Correction term
            g_dot_psi = g_i.inner(self.psi)
            correction = (sigma_sq - 1.0) * self.lambda_psi * g_dot_psi ** 2

            var_i = var_C + correction
            std_i = np.sqrt(max(0.0, var_i))
            self.component_stds.append(std_i)

        # Compute CVaR of the Gaussian mixture
        self.var, self.cvar = gaussian_mixture_cvar(
            self.component_means,
            self.component_stds,
            self.component_weights,
            self.beta,
        )

        # Find dominant component for gradient
        self.dominant_component_idx = np.argmax(self.component_weights)

        if self.verbose and self.mpi_rank == 0:
            print(f"  [Mixture Linear CVaR] N_mix={self.N_mix}, VaR={self.var:.4e}, CVaR={self.cvar:.4e}")
            for i in range(self.N_mix):
                print(f"    Component {i}: Q={self.component_means[i]:.4e}, std={self.component_stds[i]:.4e}, w={self.component_weights[i]:.4f}")

        return self.cvar

    def costValue(self, z):
        """Evaluate cost at control z."""
        self.func_ncalls += 1
        self.z.zero()
        self.z.axpy(1.0, z)
        objective = self.objective()
        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalty

    def costGradient(self, z):
        """Compute gradient of cost at control z.

        Uses simplified gradient through the dominant mixture component.
        """
        self.grad_ncalls += 1
        self.z.zero()
        self.z.axpy(1.0, z)
        self.objective()  # Ensure components are computed

        dz = self.model.generate_vector(CONTROL)

        # Simplified gradient: use the dominant component's gradient
        # More sophisticated: weight by component contributions to CVaR
        i = self.dominant_component_idx
        m_i = self.m_bar_i[i]

        # Solve forward and adjoint at dominant component
        x_i = self.model.generate_vector(STATE)
        y_i = self.model.generate_vector(STATE)
        x_all_i = [x_i, m_i, y_i, self.z]

        self.pde.solveFwd(x_i, x_all_i)
        x_all_i[STATE] = x_i

        rhs_i = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(x_all_i, rhs_i)
        self.pde.solveAdj(y_i, x_all_i, rhs_i)
        x_all_i[ADJOINT] = y_i

        self.pde.setLinearizationPoint(x_all_i, False)

        # Gradient w.r.t. control
        self.pde.evalGradientControl(x_all_i, dz)

        # Add penalization gradient
        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        norm = np.sqrt(dz.inner(dz))
        self.grad_cache = dz.copy()
        return dz, norm


class TaylorMixtureLinearCVaRControlCostFunctional(ControlCostFunctional):
    """Gaussian mixture with linear Taylor CVaR approximation.

    Uses analytical Gaussian CVaR formula for each mixture component.
    The mixture decomposes the prior along a dominant direction (KLE or HEP).
    """

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_mixture_linear_cvar_settings(settings)
        self._legacy = _TaylorMixtureLinearCVaRLegacy(
            self.settings, control_model, prior, penalization, tol
        )

    @property
    def cvar(self):
        """Return the CVaR value."""
        return self._legacy.cvar

    @property
    def var(self):
        """Return the VaR (quantile) value."""
        return self._legacy.var

    @property
    def component_means(self):
        """Return QoI values at component means."""
        return self._legacy.component_means

    @property
    def component_stds(self):
        """Return standard deviations of linear Taylor at each component."""
        return self._legacy.component_stds

    @property
    def component_weights(self):
        """Return mixture weights."""
        return self._legacy.component_weights

    def generate_vector(self, component="ALL"):
        return self._legacy.model.generate_vector(component)

    def cost(self, z, order=0):
        value = self._legacy.costValue(z)
        if order >= 1:
            self._legacy.costGradient(z)
        return value

    def grad(self, g):
        if self._legacy.grad_cache is None:
            dz, _ = self._legacy.costGradient(self._legacy.z)
        else:
            dz = self._legacy.grad_cache
        g.zero()
        g.axpy(1.0, dz)
        self._legacy.grad_cache = None
        return np.sqrt(g.inner(g))


__all__ = [
    "TaylorMixtureLinearCVaRControlCostFunctional",
    "gaussian_mixture_cvar",
]
