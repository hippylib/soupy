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
from mpi4py import MPI
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
from .linear_cvar import gaussian_cvar, gaussian_cvar_grad_std


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
    """Implementation of Gaussian-mixture linear Taylor CVaR."""

    def __init__(self, settings, model, prior, penalization, tol=1e-9):
        self.settings = settings
        self.model = model
        self.pde = model.problem
        self.qoi = model.qoi
        self.prior = prior
        self.penalization = penalization
        self.tol = tol

        self.z = model.generate_vector(CONTROL)
        self.z_diff = model.generate_vector(CONTROL)
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
        self.component_vars = []  # <g_i, C_i @ g_i>
        self.component_weights = self.mix_1d['weights']

        self.x_components = []
        self.y_components = []
        self.Cdmq_components = []

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
        self.mixture_mean = 0.0
        self.mixture_std = 0.0

    def _z_has_changed(self, z):
        """Return True when the incoming control differs from the cached one."""
        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            z_local = z[idx[0]:idx[1]]
            delta = self.z.get_local() - z_local
            return np.dot(delta, delta) > 1e-20

        self.z_diff.zero()
        self.z_diff.axpy(1.0, self.z)
        self.z_diff.axpy(-1.0, z)
        return self.z_diff.inner(self.z_diff) > 1e-20

    def _std_gradient_scale(self):
        """Return kappa / Sigma for J = M + kappa * Sigma."""
        if self.mixture_std <= 1e-14:
            return 0.0
        return gaussian_cvar_grad_std(self.mixture_std, self.beta) / self.mixture_std

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
            # Compute dominant HEP eigenvector on rank 0 and broadcast it.
            world_comm = MPI.COMM_WORLD
            if world_comm.rank == 0:
                omega = MultiVector(self.pde.generate_parameter(), 15)
                rand = Random()
                for i in range(15):
                    rand.normal(1.0, omega[i])

                d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, 1, s=1)
                psi_local = U[0].get_local()
                lambda_psi = 1.0  # U is already C^{-1}-orthonormal
                dominant_eigenvalue = float(d[0])
            else:
                psi_local = None
                lambda_psi = None
                dominant_eigenvalue = None

            psi_local = world_comm.bcast(psi_local, root=0)
            lambda_psi = world_comm.bcast(lambda_psi, root=0)
            dominant_eigenvalue = world_comm.bcast(dominant_eigenvalue, root=0)

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.set_local(psi_local)
            self.psi.apply("")
            self.lambda_psi = lambda_psi

            if self.verbose and world_comm.rank == 0:
                print(f"  [Mixture] Using HEP direction, dominant eigenvalue = {dominant_eigenvalue:.4e}")

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
        """Compute the GMM linear CVaR surrogate J = M + k * Sigma."""
        self._compute_direction()

        sigma_1d = self.mix_1d['sigma']
        sigma_sq = sigma_1d ** 2

        self.component_means = []
        self.component_stds = []
        self.component_vars = []
        self.x_components = []
        self.y_components = []
        self.Cdmq_components = []

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

            # Get rank-1 correction for componentwise variance
            g_dot_psi = g_i.inner(self.psi)
            correction = (sigma_sq - 1.0) * self.lambda_psi * g_dot_psi ** 2

            var_i = var_C + correction
            C_g_i.axpy((sigma_sq - 1.0) * self.lambda_psi * g_dot_psi, self.psi)
            self.component_vars.append(var_i)
            std_i = np.sqrt(max(0.0, var_i))
            self.component_stds.append(std_i)
            self.x_components.append(x_i)
            self.y_components.append(y_i)
            self.Cdmq_components.append(C_g_i)

        weights = np.asarray(self.component_weights)
        means = np.asarray(self.component_means)
        vars_ = np.asarray(self.component_vars)

        self.mixture_mean = float(np.sum(weights * means))
        second_moment = float(np.sum(weights * (means ** 2 + vars_)))
        mixture_var = max(0.0, second_moment - self.mixture_mean ** 2)
        self.mixture_std = np.sqrt(mixture_var)
        self.var = self.mixture_mean + self.mixture_std * norm.ppf(self.beta)
        self.cvar = gaussian_cvar(self.mixture_mean, self.mixture_std, self.beta)

        if self.verbose and self.mpi_rank == 0:
            print(
                f"  [Mixture Linear CVaR] N_mix={self.N_mix}, "
                f"mean={self.mixture_mean:.4e}, std={self.mixture_std:.4e}, "
                f"VaR={self.var:.4e}, CVaR={self.cvar:.4e}"
            )
            for i in range(self.N_mix):
                print(
                    f"    Component {i}: Q={self.component_means[i]:.4e}, "
                    f"std={self.component_stds[i]:.4e}, w={self.component_weights[i]:.4f}"
                )

        return self.cvar

    def costValue(self, z):
        """Evaluate cost at control z."""
        self.func_ncalls += 1
        z_changed = self._z_has_changed(z)
        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
            self.z.apply("")
        else:
            self.z.zero()
            self.z.axpy(1.0, z)
        if z_changed:
            self.direction_computed = False
        objective = self.objective()
        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        self.grad_cache = None
        return objective + penalty

    def _component_weighted_gradient(self, x_i, y_i, m_i, Cdmq_i, gamma_i, eta_i):
        """Compute one component contribution to the GMM linear CVaR gradient."""
        Vh = self.pde.Vh

        x_fun = vector2Function(x_i, Vh[STATE])
        y_fun = vector2Function(y_i, Vh[ADJOINT])
        m_fun = vector2Function(m_i, Vh[PARAMETER])
        z_fun = vector2Function(self.z, Vh[CONTROL])
        Cdmq_fun = vector2Function(Cdmq_i, Vh[PARAMETER])

        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        x_test = dl.TestFunction(Vh[STATE])
        y_test = dl.TestFunction(Vh[ADJOINT])
        z_test = dl.TestFunction(Vh[CONTROL])

        dmrC = dl.derivative(form, m_fun, Cdmq_fun)

        xstarrhs = self.pde.generate_state()
        if eta_i != 0.0:
            dmyrC = dl.assemble(dl.derivative(dmrC, y_fun, y_test))
            [bc.apply(dmyrC) for bc in self.pde.bc0]
            xstarrhs.axpy(2.0 * eta_i, dmyrC)
        # eta_i is w_i*kappa/(2*std) here

        xstar = self.pde.generate_state()
        self.pde.solveIncremental(xstar, -xstarrhs, False)
        # For the incremental state equation, what we solve for is indeed solve for w_i * xstar instead of xstar 

        ystarrhs = self.pde.generate_state()

        if gamma_i != 0.0:
            dxq = self.pde.generate_state()
            self.qoi.grad(STATE, [x_i, m_i, y_i, self.z], dxq)
            [bc.apply(dxq) for bc in self.pde.bc0]
            ystarrhs.axpy(gamma_i, dxq)
            # gamma_i = w_i * (1.0 + (kappa/std) * (mu_i - self.mixture_mean))

        if eta_i != 0.0:
            dmxrC = dl.assemble(dl.derivative(dmrC, x_fun, x_test))
            [bc.apply(dmxrC) for bc in self.pde.bc0]
            ystarrhs.axpy(2.0 * eta_i, dmxrC)
        # eta_i is w_i*kappa/(2*std) here

        xstar_fun = vector2Function(xstar, Vh[STATE])
        dxr = dl.derivative(form, x_fun, xstar_fun)
        dxxr = dl.assemble(dl.derivative(dxr, x_fun, x_test))
        [bc.apply(dxxr) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxr)

        dxxq = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, xstar, dxxq)
        [bc.apply(dxxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxq)

        ystar = self.pde.generate_state()
        self.pde.solveIncremental(ystar, -ystarrhs, True)
        # For the incremental state equation, what we solve for is indeed solve for w_i * ystar instead of ystar 

        grad = self.model.generate_vector(CONTROL)

        if eta_i != 0.0:
            dmzrC = dl.assemble(dl.derivative(dmrC, z_fun, z_test))
            grad.axpy(2.0 * eta_i, dmzrC)
        # eta_i is w_i*kappa/(2*std) here

        ystar_fun = vector2Function(ystar, Vh[ADJOINT])
        dyr = dl.derivative(form, y_fun, ystar_fun)
        dyzr = dl.assemble(dl.derivative(dyr, z_fun, z_test))
        grad.axpy(1.0, dyzr)

        dxzr = dl.assemble(dl.derivative(dxr, z_fun, z_test))
        grad.axpy(1.0, dxzr) # we don't need to multiply w_i here because the ystar_fun and xstar_fun here is indeed w_i * ystar and w_i * xstar, 
        # so the w_i is already included in the solution of the incremental state and adjoint

        return grad

    def costGradient(self, z):
        """Compute gradient of cost at control z."""
        self.grad_ncalls += 1
        if isinstance(z, np.ndarray):
            self.costValue(z)
        else:
            self.z_diff.zero()
            self.z_diff.axpy(1.0, self.z)
            self.z_diff.axpy(-1.0, z)
            if (not self.direction_computed) or self.z_diff.inner(self.z_diff) > 1e-20:
                self.costValue(z)

        dz = self.model.generate_vector(CONTROL)
        scale = self._std_gradient_scale() # This is kappa / std

        for i in range(self.N_mix):
            m_i = self.m_bar_i[i]
            x_i = self.x_components[i]
            y_i = self.y_components[i]
            Cdmq_i = self.Cdmq_components[i]

            x_all_i = [x_i, m_i, y_i, self.z]
            self.pde.setLinearizationPoint(x_all_i, False)
            self.qoi.setLinearizationPoint(x_all_i)

            w_i = self.component_weights[i]
            mu_i = self.component_means[i]

            gamma_i = w_i * (1.0 + scale * (mu_i - self.mixture_mean))
            eta_i = 0.5 * scale * w_i

            dcomp_i = self._component_weighted_gradient(
                x_i, y_i, m_i, Cdmq_i, gamma_i, eta_i
            )
            dz.axpy(1.0, dcomp_i)

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

    The objective follows the GMM linear surrogate
        J = M + k * Sigma + P(z),
    where M and Sigma are assembled from the mixture component means and
    linearized component variances.
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
