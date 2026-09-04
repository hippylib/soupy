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

from ...collectives.collective import MultipleSerialPDEsCollective, NullCollective
from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL
from .settings import taylor_mixture_linear_cvar_settings
from .gmm_library import get_1d_gmm_library_mixture
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


class _TaylorAlternatingHEPLinearCVaRLegacy:
    """Implementation of Gaussian-mixture linear Taylor CVaR with fixed HEP direction per instance."""

    def __init__(self, settings, model, prior, penalization, tol=1e-9, comm_sampler=None):
        self.settings = settings
        self.model = model
        self.pde = model.problem
        self.qoi = model.qoi
        self.prior = prior
        self.penalization = penalization
        self.tol = tol

        self.z = model.generate_vector(CONTROL)
        self.z_diff = model.generate_vector(CONTROL)
        self.z_at_direction = model.generate_vector(CONTROL)
        self.direction_diff = model.generate_vector(CONTROL)
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
        self.seed = settings["seed"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        # MPI setup
        self.mesh_comm = self.pde.Vh[STATE].mesh().mpi_comm()
        self.mesh_rank = dl.MPI.rank(self.mesh_comm)
        self.mesh_size = dl.MPI.size(self.mesh_comm)
        self.mix_comm = MPI.COMM_WORLD if comm_sampler is None else comm_sampler
        self.mix_rank = self.mix_comm.Get_rank()
        self.mix_size = self.mix_comm.Get_size()
        self.cluster_parallel = self.mesh_size == 1 and self.mix_size > 1
        self.collective = (
            MultipleSerialPDEsCollective(self.mix_comm)
            if self.cluster_parallel
            else NullCollective()
        )
        self.mpi_rank = self.mix_rank if self.cluster_parallel else self.mesh_rank
        self.mpi_size = self.mesh_size

        # Get 1D mixture parameters
        self.mix_1d = get_1d_gmm_library_mixture(
            self.N_mix, rule=1, warn=(self.mpi_rank == 0)
        )

        # Storage for mixture component quantities
        self.component_means = []  # Q(m_bar_i)
        self.component_stds = []  # sqrt(<g_i, C_i @ g_i>)
        self.component_vars = []  # <g_i, C_i @ g_i>
        self.component_weights = self.mix_1d['weights']

        self.x_components = []
        self.y_components = []
        self.Cdmq_components = []
        self.local_component_results = {}

        # Direction and components (computed on first objective call)
        self.psi = None  # Decomposition direction
        self.lambda_psi = None  # Pseudo-eigenvalue
        self.m_bar_i = []  # Component means in parameter space
        self.direction_computed = False
        self.direction_is_current = False

        # Hessian for HEP direction
        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        # CVaR values
        self.var = 0.0
        self.cvar = 0.0
        self.mixture_mean = 0.0
        self.mixture_std = 0.0

    def _is_print_root(self):
        return self.mix_rank == 0 if self.cluster_parallel else self.mesh_rank == 0

    def _owned_component_indices(self):
        if not self.cluster_parallel:
            return tuple(range(self.N_mix))
        return tuple(i for i in range(self.N_mix) if i % self.mix_size == self.mix_rank)

    def _reset_component_storage(self):
        self.component_means = [None] * self.N_mix
        self.component_stds = [None] * self.N_mix
        self.component_vars = [None] * self.N_mix
        self.x_components = {}
        self.y_components = {}
        self.Cdmq_components = {}
        self.local_component_results = {}

    def _store_component_result(self, index, result):
        self.local_component_results[index] = result
        self.component_means[index] = result["mean"]
        self.component_stds[index] = result["std"]
        self.component_vars[index] = result["var"]
        self.x_components[index] = result["x"]
        self.y_components[index] = result["y"]
        self.Cdmq_components[index] = result["Cdmq"]

    def _gather_component_summaries(self):
        local_payload = []
        for index, comp in self.local_component_results.items():
            local_payload.append(
                {
                    "index": index,
                    "mean": float(comp["mean"]),
                    "std": float(comp["std"]),
                    "var": float(comp["var"]),
                }
            )

        for payloads in self.mix_comm.allgather(local_payload):
            for payload in payloads:
                index = payload["index"]
                self.component_means[index] = payload["mean"]
                self.component_stds[index] = payload["std"]
                self.component_vars[index] = payload["var"]

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

    def _cache_direction_z(self):
        self.z_at_direction.zero()
        self.z_at_direction.axpy(1.0, self.z)
        self.direction_is_current = True

    def _direction_matches_current_z(self):
        if not self.direction_is_current:
            return False
        self.direction_diff.zero()
        self.direction_diff.axpy(1.0, self.z_at_direction)
        self.direction_diff.axpy(-1.0, self.z)
        return self.direction_diff.inner(self.direction_diff) <= 1e-20

    def _compute_direction(self, FD_gradient_check=False):
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
            if self.mix_rank == 0:
                omega = MultiVector(self.pde.generate_parameter(), 64)
                rand = Random(seed=self.seed)
                for i in range(64):
                    rand.normal(1.0, omega[i])

                d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, omega.nvec(), s=1)
                dominant_idx = int(np.argmax(np.abs(d)))
                psi_local = U[dominant_idx].get_local()
                lambda_psi = 1.0  # U is already C^{-1}-orthonormal
                dominant_eigenvalue = float(d[dominant_idx])
            else:
                psi_local = None
                lambda_psi = None
                dominant_eigenvalue = None

            psi_local = self.mix_comm.bcast(psi_local, root=0)
            lambda_psi = self.mix_comm.bcast(lambda_psi, root=0)
            dominant_eigenvalue = self.mix_comm.bcast(dominant_eigenvalue, root=0)

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.set_local(psi_local)
            self.psi.apply("")
            self.lambda_psi = lambda_psi

            if self.verbose and self._is_print_root():
                print(f"  [Mixture] Using HEP direction, dominant eigenvalue = {dominant_eigenvalue:.4e}")

        else:  # KLE direction
            # Use power iteration to get dominant KLE eigenvector
            v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            rand = Random(seed=self.seed)
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

            if self.verbose and self._is_print_root():
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
        self._cache_direction_z()

    def objective(self, FD_gradient_check=False):
        """Compute the GMM linear CVaR surrogate J = M + k * Sigma."""
        self._compute_direction(FD_gradient_check=FD_gradient_check)

        sigma_1d = self.mix_1d['sigma']
        sigma_sq = sigma_1d ** 2

        self._reset_component_storage()
        local_mean = 0.0
        local_second_moment = 0.0

        # Pre-compute R @ psi = C^{-1} @ psi
        R_psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
        self.prior.R.mult(self.psi, R_psi)

        for i in self._owned_component_indices():
            m_i = self.m_bar_i[i]

            # Solve forward at component mean m_i
            x_i = self.model.generate_vector(STATE)
            y_i = self.model.generate_vector(STATE)
            x_all_i = [x_i, m_i, y_i, self.z]

            self.pde.solveFwd(x_i, x_all_i)
            x_all_i[STATE] = x_i

            # Compute QoI at component mean
            Q_i = self.qoi.cost(x_all_i)
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
            std_i = np.sqrt(max(0.0, var_i))
            self._store_component_result(
                i,
                {
                    "mean": float(Q_i),
                    "std": float(std_i),
                    "var": float(var_i),
                    "x": x_i,
                    "y": y_i,
                    "Cdmq": C_g_i,
                },
            )
            local_mean += self.component_weights[i] * float(Q_i)
            local_second_moment += self.component_weights[i] * float(Q_i ** 2 + var_i)

        weights = np.asarray(self.component_weights)
        if self.cluster_parallel:
            self._gather_component_summaries()
            self.mixture_mean = float(self.collective.allReduce(local_mean, "sum"))
            second_moment = float(self.collective.allReduce(local_second_moment, "sum"))
        else:
            means = np.asarray(self.component_means)
            vars_ = np.asarray(self.component_vars)
            self.mixture_mean = float(np.sum(weights * means))
            second_moment = float(np.sum(weights * (means ** 2 + vars_)))

        means = np.asarray(self.component_means)
        mixture_var = max(0.0, second_moment - self.mixture_mean ** 2)
        self.mixture_std = np.sqrt(mixture_var)
        self.var, self.cvar = gaussian_mixture_cvar(
            means,
            self.component_stds,
            weights,
            self.beta,
        )

        if self.verbose and self._is_print_root():
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

    def costValue(self, z, FD_gradient_check=False):
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

        # Keep the HEP direction fixed for the lifetime of this objective.
        if z_changed and self.direction != "hep" and not FD_gradient_check:
            self.direction_computed = False
            self.direction_is_current = False
        
        objective = self.objective(FD_gradient_check=FD_gradient_check)
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

        xstar = self.pde.generate_state()
        self.pde.solveIncremental(xstar, -xstarrhs, False)
        # The weighted factor is already embedded in xstar through xstarrhs.

        ystarrhs = self.pde.generate_state()

        if gamma_i != 0.0:
            dxq = self.pde.generate_state()
            self.qoi.grad(STATE, [x_i, m_i, y_i, self.z], dxq)
            [bc.apply(dxq) for bc in self.pde.bc0]
            ystarrhs.axpy(gamma_i, dxq)

        if eta_i != 0.0:
            dmxrC = dl.assemble(dl.derivative(dmrC, x_fun, x_test))
            [bc.apply(dmxrC) for bc in self.pde.bc0]
            ystarrhs.axpy(2.0 * eta_i, dmxrC)

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

        ystar_fun = vector2Function(ystar, Vh[ADJOINT])
        dyr = dl.derivative(form, y_fun, ystar_fun)
        dyzr = dl.assemble(dl.derivative(dyr, z_fun, z_test))
        grad.axpy(1.0, dyzr)

        dxzr = dl.assemble(dl.derivative(dxr, z_fun, z_test))
        grad.axpy(1.0, dxzr)

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
        denom = max(1.0 - self.beta, 1e-14)

        for i in self._owned_component_indices():
            m_i = self.m_bar_i[i]
            x_i = self.x_components[i]
            y_i = self.y_components[i]
            Cdmq_i = self.Cdmq_components[i]

            x_all_i = [x_i, m_i, y_i, self.z]
            self.pde.setLinearizationPoint(x_all_i, False)
            self.qoi.setLinearizationPoint(x_all_i)

            w_i = self.component_weights[i]
            mu_i = self.component_means[i]
            sigma_i = self.component_stds[i]

            if sigma_i > 1e-14:
                z_i = (self.var - mu_i) / sigma_i
                exceedance_i = 1.0 - norm.cdf(z_i)
                gamma_i = w_i * exceedance_i / denom
                eta_i = 0.5 * w_i * norm.pdf(z_i) / (sigma_i * denom)
            else:
                exceedance_i = 1.0 if mu_i > self.var else 0.0
                gamma_i = w_i * exceedance_i / denom
                eta_i = 0.0

            dcomp_i = self._component_weighted_gradient(
                x_i, y_i, m_i, Cdmq_i, gamma_i, eta_i
            )
            dz.axpy(1.0, dcomp_i)

        if self.cluster_parallel:
            self.collective.allReduce(dz, "sum")

        # Add penalization gradient
        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        grad_norm = np.sqrt(dz.inner(dz))
        self.grad_cache = dz.copy()
        return dz, grad_norm


class TaylorAlternatingHEPLinearCVaRControlCostFunctional(ControlCostFunctional):
    """Gaussian mixture with linear Taylor CVaR approximation and fixed HEP direction.

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
        comm_sampler=None,
    ):
        self.settings = taylor_mixture_linear_cvar_settings(settings)
        self._legacy = _TaylorAlternatingHEPLinearCVaRLegacy(
            self.settings, control_model, prior, penalization, tol, comm_sampler=comm_sampler
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

    @property
    def cluster_parallel_enabled(self):
        return self._legacy.cluster_parallel

    @property
    def owned_component_indices(self):
        return self._legacy._owned_component_indices()

    def generate_vector(self, component="ALL"):
        return self._legacy.model.generate_vector(component)

    def cost(self, z, order=0, FD_gradient_check=False):
        value = self._legacy.costValue(z, FD_gradient_check=FD_gradient_check)
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
    "TaylorAlternatingHEPLinearCVaRControlCostFunctional",
    "gaussian_mixture_cvar",
]
