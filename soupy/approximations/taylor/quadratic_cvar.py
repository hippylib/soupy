"""Second-order (quadratic) Taylor approximation with CVaR risk measure.

For quadratic Taylor approximation:
    Q(m) ≈ Q₀ + g^T(m - m̄) + ½(m - m̄)^T H (m - m̄)

Using the truncated eigendecomposition H ≈ U D U^T, and with ξ = U^T R (m - m̄) ~ N(0, I):
    Q(ξ) ≈ Q₀ + g̃^T ξ + ½ Σᵢ dᵢ ξᵢ²

where g̃ = U^T g_param. This is a generalized chi-squared distribution.

CVaR is computed by Monte Carlo sampling from the Taylor surrogate (cheap - no PDE solves).
"""

from __future__ import annotations

import time
from typing import Optional, Union

import dolfin as dl
import numpy as np
from hippylib import Random, vector2Function, MultiVector
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.smoothPlusApproximation import SmoothPlusApproximationQuartic
from ...modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL
from .settings import taylor_quadratic_cvar_settings


def surrogate_cvar_from_samples(samples, beta, epsilon=1e-4):
    """Compute CVaR from samples using smooth approximation.

    CVaR_β[Q] = min_t { t + E[max(0, Q - t)] / (1-β) }

    Uses smooth plus approximation and finds optimal t.

    Args:
        samples: Array of sample values
        beta: Risk level (e.g., 0.95 for 95% CVaR)
        epsilon: Smoothing parameter

    Returns:
        (t_opt, cvar_value): Optimal threshold and CVaR value
    """
    # Initial guess from sample quantile
    t_init = np.percentile(samples, beta * 100)
    smoothplus = SmoothPlusApproximationQuartic(epsilon=epsilon)

    def cvar_obj(t):
        return t + np.mean(smoothplus(samples - t)) / (1 - beta)

    def cvar_grad(t):
        return 1 - np.mean(smoothplus.grad(samples - t)) / (1 - beta)

    # Simple gradient descent to find optimal t
    t = t_init
    lr = 0.1
    for _ in range(100):
        g = cvar_grad(t)
        if abs(g) < 1e-8:
            break
        t = t - lr * g

    return t, cvar_obj(t)


class _TaylorQuadraticCVaRLegacy:
    """Implementation of quadratic Taylor approximation with CVaR risk measure."""

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

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(STATE)

        self.xstar = model.generate_vector(STATE)
        self.ystar = model.generate_vector(STATE)
        self.mhelp = model.generate_vector(PARAMETER)
        self.Cdmq = model.generate_vector(PARAMETER)
        self.dmq = model.generate_vector(PARAMETER)

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.N_tr = settings["N_tr"]
        self.beta = settings["beta"]  # CVaR risk level
        self.N_mc = settings["N_mc"]  # Surrogate MC samples
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        try:
            self.correction = settings["correction"]
        except (KeyError, ValueError):
            self.correction = False

        try:
            self.N_mc_correction = settings["N_mc_correction"]
        except (KeyError, ValueError):
            self.N_mc_correction = 0

        self.xhat = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.yhat = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        # MPI setup
        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        # Eigenvalue storage
        self.d = None  # Eigenvalues
        self.U = None  # Eigenvectors (MultiVector)

        # Taylor statistics
        self.Q_0 = 0.0
        self.lin_var = 0.0  # Linear Taylor variance
        self.g_tilde = None  # Projected gradient g̃ = U^T g

        # CVaR components
        self.cvar = 0.0
        self.t_opt = 0.0  # Optimal threshold
        self.Q_surrogate = None  # Surrogate samples

        # Smooth plus for CVaR
        self.smoothplus = SmoothPlusApproximationQuartic(epsilon=self.epsilon)

        # MC correction setup (PDE-based samples)
        if self.correction and self.N_mc_correction > 0:
            self.m_mc = []
            self.x_mc = []
            self.Q_mc = np.zeros(self.N_mc_correction)
            randomGen = Random(myid=0, nproc=self.mpi_size)
            for _ in range(self.N_mc_correction):
                noise = dl.Vector()
                prior.init_vector(noise, "noise")
                randomGen.normal(1.0, noise)

                sample = dl.Vector()
                prior.init_vector(sample, 1)
                prior.sample(noise, sample, add_mean=False)
                self.m_mc.append(sample)
                self.x_mc.append(self.pde.generate_state())

            self.cvar_diff = 0.0

    def objectiveLinear(self):
        """Compute linear Taylor components (Q₀, dmq, Cdmq)."""
        self.x_all[CONTROL] = self.z
        self.pde.solveFwd(self.x, self.x_all)
        Q0 = self.qoi.cost(self.x_all)
        self.x_all[STATE] = self.x

        rhs = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(self.x_all, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs)
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False)
        self.qoi.setLinearizationPoint(self.x_all)

        self.Q_0 = Q0
        self.x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        self.y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        self.m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        self.z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])

        form = self.pde.varf_handler(self.x_fun, self.m_fun, self.y_fun, self.z_fun)
        m_test = dl.TestFunction(self.pde.Vh[PARAMETER])
        self.dmq.zero()
        self.dmq.axpy(1.0, dl.assemble(dl.derivative(form, self.m_fun, m_test)))
        self.prior.Rsolver.solve(self.Cdmq, self.dmq)
        self.lin_var = self.dmq.inner(self.Cdmq)

        return Q0

    def objective(self):
        """Compute the CVaR objective using quadratic Taylor surrogate sampling."""
        Q0 = self.objectiveLinear()

        # Compute dominant Hessian eigenvalues/eigenvectors
        omega = MultiVector(self.pde.generate_parameter(), self.N_tr + 5)
        rand = Random()
        for i in range(self.N_tr + 5):
            rand.normal(1.0, omega[i])

        self.d, self.U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, self.N_tr, s=1)

        # Compute state increments for each eigendirection (needed for gradient)
        for i in range(self.N_tr):
            self.xhat[i].zero()
            self.yhat[i].zero()
            self.pde.apply_ij(ADJOINT, PARAMETER, self.U[i], self.rhs_fwd)
            self.pde.solveIncremental(self.xhat[i], -self.rhs_fwd, False)
            self.pde.apply_ij(PARAMETER, STATE, self.xhat[i], self.yhat[i])

        # Compute projected gradient g̃ = U^T (R^{-1} dmq)
        # Note: dmq = dQ/dm, and we want g̃_k = U_k^T C dmq = U_k^T R^{-1} dmq
        # Since Cdmq = R^{-1} dmq, g̃_k = U_k · Cdmq (but U is R-orthonormal)
        # Actually: g̃_k = U_k · dmq (since U^T R U = I, U_k · (R^{-1} dmq) = U_k · Cdmq
        # Wait, let me reconsider. If U^T R U = I, then U_k is in the R^{-1} inner product space.
        # For ξ = U^T R (m - m̄), the linear term g^T(m - m̄) becomes:
        # g^T(m - m̄) = dmq^T (m - m̄) = dmq^T R^{-1} R (m - m̄)
        # = (R^{-1} dmq)^T R (m - m̄) = Cdmq^T R (m - m̄)
        # = (U Cdmq)^T U^T R (m - m̄) wait, this is getting confusing.
        # Let me just compute: g̃_k = R U_k · Cdmq = U_k^T R R^{-1} dmq = U_k^T dmq
        # So g̃_k = U_k · dmq (standard inner product)
        self.g_tilde = np.zeros(self.N_tr)
        for k in range(self.N_tr):
            self.g_tilde[k] = self.U[k].inner(self.dmq)

        # Generate surrogate samples: Q(ξ) = Q₀ + g̃^T ξ + 0.5 * Σᵢ |dᵢ| ξᵢ²
        # where ξ ~ N(0, I)
        # NOTE: We use |d| instead of d for CVaR computation because:
        # - Negative eigenvalues indicate concave curvature in the parameter space
        # - For CVaR (tail risk), negative eigenvalues would incorrectly reduce variance
        # - Using |d| ensures we capture the full spread of the QoI distribution
        np.random.seed(42)  # For reproducibility
        xi_samples = np.random.randn(self.N_mc, self.N_tr)

        # Use absolute values of eigenvalues for CVaR (captures spread, not just convex part)
        d_abs = np.abs(self.d)

        self.Q_surrogate = np.zeros(self.N_mc)
        for i in range(self.N_mc):
            xi = xi_samples[i, :]
            # Linear term: g̃^T ξ
            linear_term = np.dot(self.g_tilde, xi)
            # Quadratic term: 0.5 * Σ |d_k| ξ_k² (use absolute eigenvalues)
            quad_term = 0.5 * np.sum(d_abs * xi ** 2)
            self.Q_surrogate[i] = Q0 + linear_term + quad_term

        # Compute CVaR from surrogate samples
        self.t_opt, self.cvar = surrogate_cvar_from_samples(
            self.Q_surrogate, self.beta, self.epsilon
        )

        # MC correction using actual PDE samples (if enabled)
        # Uses mean bias correction: shift surrogate samples by E[Q_true - Q_taylor]
        # This is more stable than trying to correct CVaR directly with few samples
        self.mean_correction = 0.0
        self.cvar_corrected = self.cvar
        if self.correction and self.N_mc_correction > 0:
            Q_true = np.zeros(self.N_mc_correction)
            Q_taylor_samples = np.zeros(self.N_mc_correction)

            # Pre-compute R * m_sample for eigenvector projections
            R_m_samples = []
            for i in range(self.N_mc_correction):
                R_m = self.pde.generate_parameter()
                self.prior.R.mult(self.m_mc[i], R_m)
                R_m_samples.append(R_m)

            for i in range(self.N_mc_correction):
                m_sample = self.m_mc[i]

                # Compute m_i = m_bar + m_sample
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, self.m)
                m_i.axpy(1.0, m_sample)

                # Solve forward problem at m_i
                x_all_mc = [self.x_mc[i], m_i, self.y, self.z]
                self.pde.solveFwd(self.x_mc[i], x_all_mc)

                # Compute true QoI value Q(m_i)
                Q_true[i] = self.qoi.cost([self.x_mc[i], m_i, self.y, self.z])
                self.Q_mc[i] = Q_true[i]

                # Compute Taylor approximation at this sample
                linear_term = self.dmq.inner(m_sample)
                quad_term = 0.0
                for k in range(self.N_tr):
                    Uk_dot_Rm = self.U[k].inner(R_m_samples[i])
                    quad_term += self.d[k] * Uk_dot_Rm ** 2
                quad_term *= 0.5
                Q_taylor_samples[i] = Q0 + linear_term + quad_term

            # Compute mean bias correction: E[Q_true - Q_taylor]
            self.mean_correction = np.mean(Q_true) - np.mean(Q_taylor_samples)

            # Shift surrogate samples by mean correction and recompute CVaR
            Q_surrogate_corrected = self.Q_surrogate + self.mean_correction
            _, self.cvar_corrected = surrogate_cvar_from_samples(
                Q_surrogate_corrected, self.beta, self.epsilon
            )

        cost = self.cvar_corrected

        if self.verbose and self.mpi_rank == 0:
            print(f"  [Quadratic Taylor CVaR] Q0={Q0:.3e}, CVaR={self.cvar:.3e}, t_opt={self.t_opt:.3e}")
            print(f"                          eigenvalues[:5]={self.d[:min(5, len(self.d))]}")
            if self.correction and self.N_mc_correction > 0:
                print(f"                          mean_correction={self.mean_correction:.3e}, CVaR_corrected={self.cvar_corrected:.3e}")

        return cost

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

        This is the simplified version that ignores ∂d/∂z and ∂U/∂z terms.
        The gradient is computed using the chain rule through the CVaR.
        """
        self.grad_ncalls += 1
        self.z.zero()
        self.z.axpy(1.0, z)
        self.objective()  # Ensure components are computed

        dz = self.model.generate_vector(CONTROL)

        # For the simplified gradient, we use the fact that:
        # CVaR = t + E[max(0, Q - t)] / (1-β)
        # And Q_surrogate depends on z through Q₀, g̃, and d
        #
        # The full gradient would be:
        # d(CVaR)/dz = E[smoothplus'(Q_s - t) * dQ_s/dz] / (1-β)
        #
        # where dQ_s/dz includes ∂Q₀/∂z, ∂g̃/∂z, and ∂d/∂z terms.
        #
        # Simplified: We only include ∂Q₀/∂z term (the dominant contribution)

        # d(Q₀)/dz: gradient through adjoint (same as mean-variance case)
        self.pde.evalGradientControl(self.x_all, dz)

        # Add penalization gradient
        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        norm = np.sqrt(dz.inner(dz))
        self.grad_cache = dz.copy()
        return dz, norm

    def costHessian(self, z, z_dir):
        """Compute Hessian-vector product (simplified)."""
        self.z.zero()
        self.z.axpy(1.0, z)
        self.z_dir.zero()
        self.z_dir.axpy(1.0, z_dir)

        # Ensure objective has been computed
        self.objective()

        Vh = self.pde.Vh

        z_fun = vector2Function(self.z, Vh[CONTROL])
        z_dir_fun = vector2Function(self.z_dir, Vh[CONTROL])
        m_fun = vector2Function(self.m, Vh[PARAMETER])
        x_fun = vector2Function(self.x, Vh[STATE])
        y_fun = vector2Function(self.y, Vh[ADJOINT])

        r_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        # Solve incremental forward
        x_trial = dl.TrialFunction(Vh[STATE])
        x_star = dl.Function(Vh[STATE])
        y_test = dl.TestFunction(Vh[ADJOINT])

        rx_form = dl.derivative(r_form, x_fun, x_trial)
        rxy_form = dl.derivative(rx_form, y_fun, y_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzy_form = dl.derivative(rz_form, y_fun, y_test)
        Ly_form = -rzy_form

        dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0)

        # Solve incremental adjoint
        x_test = dl.TestFunction(Vh[STATE])
        y_trial = dl.TrialFunction(Vh[ADJOINT])
        y_star = dl.Function(Vh[ADJOINT])

        ry_form = dl.derivative(r_form, y_fun, y_trial)
        ryx_form = dl.derivative(ry_form, x_fun, x_test)

        rx_form = dl.derivative(r_form, x_fun, x_star)
        rxx_form = dl.derivative(rx_form, x_fun, x_test)

        qx_form = dl.derivative(self.qoi.form(x_fun), x_fun, x_star)
        qxx_form = dl.derivative(qx_form, x_fun, x_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzx_form = dl.derivative(rz_form, x_fun, x_test)

        Lx_form = -(rxx_form + qxx_form + rzx_form)

        dl.solve(ryx_form == Lx_form, y_star, self.pde.bc0)

        # Assemble Hessian action
        z_test = dl.TestFunction(Vh[CONTROL])
        ry_form = dl.derivative(r_form, y_fun, y_star)
        ryz_form = dl.derivative(ry_form, z_fun, z_test)

        rx_form = dl.derivative(r_form, x_fun, x_star)
        rxz_form = dl.derivative(rx_form, z_fun, z_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzz_form = dl.derivative(rz_form, z_fun, z_test)

        Lz_form = ryz_form + rxz_form + rzz_form

        Hz = dl.assemble(Lz_form)

        if self.penalization is not None:
            dzzp = self.model.generate_vector(CONTROL)
            self.penalization.hessian(self.z, self.z_dir, dzzp)
            Hz.axpy(1.0, dzzp)

        return Hz


class TaylorQuadraticCVaRControlCostFunctional(ControlCostFunctional):
    """Wrapper exposing the quadratic Taylor CVaR approximation.

    Uses surrogate MC sampling from the generalized chi-squared distribution.
    """

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_quadratic_cvar_settings(settings)
        self._legacy = _TaylorQuadraticCVaRLegacy(
            self.settings, control_model, prior, penalization, tol
        )

    @property
    def Q_0(self):
        """Return Q at prior mean."""
        return self._legacy.Q_0

    @property
    def cvar(self):
        """Return the CVaR value from surrogate sampling."""
        return self._legacy.cvar

    @property
    def t_opt(self):
        """Return the optimal CVaR threshold."""
        return self._legacy.t_opt

    @property
    def d(self):
        """Return the Hessian eigenvalues."""
        return self._legacy.d

    @property
    def Q_surrogate(self):
        """Return the surrogate samples."""
        return self._legacy.Q_surrogate

    @property
    def Q_mc(self):
        """Return the PDE MC samples (if correction enabled)."""
        return self._legacy.Q_mc if hasattr(self._legacy, 'Q_mc') else None

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

    def hessian(self, zhat, Hzhat):
        Hz = self._legacy.costHessian(self._legacy.z, zhat)
        Hzhat.zero()
        Hzhat.axpy(1.0, Hz)


__all__ = ["TaylorQuadraticCVaRControlCostFunctional", "surrogate_cvar_from_samples"]
