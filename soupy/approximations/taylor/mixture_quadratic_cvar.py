"""Gaussian-mixture quadratic Taylor approximation for CVaR.

Implementation strategy:
- Build Gaussian-mixture component means in parameter space exactly as in
  ``mixture_quadratic.py``.
- For each component mean, run the validated single-Gaussian quadratic CVaR
  pipeline from ``quadratic_cvar.py`` up to the surrogate-sample stage.
- Solve one global scalar minimization in ``t`` for the weighted mixture CVaR
  objective.
- Reuse the full adjoint-based z-gradient machinery from
  ``quadratic_cvar.py`` component-by-component, with the only change being that
  each component uses the global optimal ``t`` and the same unweighted
  single-component sample factors as ``quadratic_cvar.py``

      1 / ((1 - beta) K_i) * E_ik(t_opt),

  while the final mixture z-gradient is assembled as the weighted combination
  ``sum_i w_i * grad_i``.

This follows the user-provided theory closely while avoiding a separate
optimization variable in the control gradient.
"""

from __future__ import annotations

from typing import Optional, Union

import dolfin as dl
import numpy as np
import scipy.optimize
from hippylib import MultiVector, Random, vector2Function
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.smoothPlusApproximation import SmoothPlusApproximationQuartic
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .mixture_data import get_1d_mixture
from .mixture_quadratic import _ShiftedPrior
from .quadratic_cvar import _TaylorQuadraticCVaRLegacy
from .settings import taylor_mixture_quadratic_cvar_settings, taylor_quadratic_cvar_settings

# Provides an initial guess for the value at risk t^*
def _weighted_quantile(values, weights, q):
    """Return the weighted q-quantile for 1D samples."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.size == 0:
        raise RuntimeError("Cannot compute a weighted quantile from empty samples.")
    if values.shape != weights.shape:
        raise RuntimeError("values and weights must have the same shape.")

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    total = cumulative[-1]
    if total <= 0.0:
        raise RuntimeError("Sample weights must sum to a positive value.")

    threshold = float(np.clip(q, 0.0, 1.0)) * total
    idx = int(np.searchsorted(cumulative, threshold, side="left"))
    idx = min(idx, values_sorted.size - 1)
    return float(values_sorted[idx])


class _TaylorMixtureQuadraticCVaRLegacy:
    """Gaussian-mixture quadratic Taylor CVaR with adjoint-based z-gradient."""

    def __init__(self, settings, model, prior, penalization, tol=1e-9):
        self.settings = settings
        self.model = model
        self.pde = model.problem
        self.qoi = model.qoi
        self.prior = prior
        self.penalization = penalization
        self.tol = tol

        self.z = model.generate_vector(CONTROL)
        self.z_at_objective = model.generate_vector(CONTROL)
        self.z_diff = model.generate_vector(CONTROL)
        self.objective_is_current = False
        self.z_at_direction = model.generate_vector(CONTROL)
        self.direction_diff = model.generate_vector(CONTROL)
        self.direction_is_current = False

        self.x = model.generate_vector(STATE)
        self.y = model.generate_vector(STATE)
        self.x_all = [self.x, prior.mean, self.y, self.z]

        self.grad_cache = None

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.beta = settings["beta"]
        self.N_mix = settings["N_mix"]
        self.direction = settings["direction"]
        self.N_tr = settings["N_tr"]
        self.N_mc = settings["N_mc"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        if self.N_mc <= 0:
            raise RuntimeError("mixture quadratic CVaR requires N_mc > 0.")

        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        self.mix_1d = get_1d_mixture(self.N_mix)
        self.component_weights = np.asarray(self.mix_1d["weights"], dtype=float)

        self.psi = None
        self.lambda_psi = None
        self.m_bar_i = []
        self.direction_computed = False

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)
        self.smoothplus = SmoothPlusApproximationQuartic(epsilon=self.epsilon)

        self.component_solvers = []
        self.component_Q0 = []
        self.component_d = []
        self.component_samples = []
        self.shared_m_mc = []
        self.shared_Rm_mc = []

        self.Q_surrogate = np.zeros(0)
        self.t_opt = 0.0
        self.cvar = 0.0

    def _copy_z(self, z):
        self.z.zero()
        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0] : idx[1]])
            self.z.apply("")
        else:
            self.z.axpy(1.0, z)
        self.objective_is_current = False

    def _cache_objective_z(self):
        self.z_at_objective.zero()
        self.z_at_objective.axpy(1.0, self.z)
        self.objective_is_current = True

    def _objective_matches_current_z(self):
        if not self.objective_is_current:
            return False
        self.z_diff.zero()
        self.z_diff.axpy(1.0, self.z_at_objective)
        self.z_diff.axpy(-1.0, self.z)
        return self.z_diff.inner(self.z_diff) <= 1e-20

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

    def _compute_direction(self):
        """Compute the dominant mixture-splitting direction (KLE or HEP)."""
        if self.direction == "hep" and self.direction_computed:
            if not self._direction_matches_current_z():
                self.direction_computed = False
                self.direction_is_current = False

        if self.direction_computed:
            return

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
            omega = MultiVector(self.pde.generate_parameter(), 15)
            rand = Random()
            for i in range(15):
                rand.normal(1.0, omega[i])

            # doublePassG returns eigenvectors U satisfying U^T R U = I, so
            # lambda_psi = <psi, R psi>^{-1} = 1 for the returned direction.
            d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, 1, s=1)
            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.zero()
            self.psi.axpy(1.0, U[0])
            self.lambda_psi = 1.0
            dominant_eigenvalue = float(d[0])

            if self.verbose and self.mpi_rank == 0:
                print(
                    f"  [Mixture Quad CVaR] Using HEP direction, "
                    f"dominant eigenvalue = {dominant_eigenvalue:.4e}"
                )
        else:
            v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            rand = Random()
            rand.normal(1.0, v)

            for _ in range(50):
                w = dl.Function(self.pde.Vh[PARAMETER]).vector()
                self.prior.Rsolver.solve(w, v)
                norm_w = np.sqrt(w.inner(w))
                if norm_w > 1e-14:
                    v.zero()
                    v.axpy(1.0 / norm_w, w)

            R_v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.prior.R.mult(v, R_v)
            norm_sq = v.inner(R_v)
            v_local = v.get_local()
            v.set_local(v_local / np.sqrt(norm_sq))
            v.apply("")

            self.psi = v
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print("  [Mixture Quad CVaR] Using KLE direction")

        mix_means_1d = self.mix_1d["means"]
        sqrt_lambda = np.sqrt(self.lambda_psi)

        self.m_bar_i = []
        for i in range(self.N_mix):
            m_i = dl.Function(self.pde.Vh[PARAMETER]).vector()
            m_i.axpy(1.0, self.prior.mean)
            m_i.axpy(mix_means_1d[i] * sqrt_lambda, self.psi)
            self.m_bar_i.append(m_i)

        self.direction_computed = True
        self._cache_direction_z()

    def _component_settings(self):
        return taylor_quadratic_cvar_settings(
            {
                "beta": self.beta,
                "N_tr": self.N_tr,
                "N_mc": self.N_mc,
                "epsilon": self.epsilon,
                "correction": False,
                "N_mc_correction": 0,
                "verbose": False,
            }
        )

    # Get mc samples for estimating cvar. The samples are from N(0, C_i) where C_i is C + (sigma^2-1)*lambda_psi*psi*psi^T, which is
    # shared by all clusters from generating clusterwise MC samples. 
    def _build_shared_component_samples(self, shifted_prior):
        """Sample perturbations with covariance C_i shared by all mixture components."""
        sigma = float(self.mix_1d["sigma"])
        random_gen = Random(myid=0, nproc=self.mpi_size)

        self.shared_m_mc = []
        self.shared_Rm_mc = []

        for _ in range(self.N_mc):
            # Generate the noise vector for sampling from the shifted prior
            noise = dl.Vector()
            self.prior.init_vector(noise, "noise")
            random_gen.normal(1.0, noise)

            # Sampling from the base prior
            base_sample = dl.Vector()
            self.prior.init_vector(base_sample, 1)
            self.prior.sample(noise, base_sample, add_mean=False)

            # To generate samples from N(0, C_i), we apply change of variables below: 
            # m_shift = m_base + (sigma - 1) * lambda_psi <psi, C^{-1} m_base> psi, 
            # which should have covariance C_i = C + (sigma^2 - 1) * lambda_psi * psi * psi^T. 
            
            # Computes <psi, C^{-1} m_base> for the covariance shift
            base_precision = dl.Vector()
            self.prior.init_vector(base_precision, 1)
            self.prior.R.mult(base_sample, base_precision) # base_precision = C^{-1} m_base

            coeff = self.psi.inner(base_precision)

            # Compute m_shift = m_base + (sigma - 1) * lambda_psi * coef * psi
            #  Here lambda_psi = <psi, C^{-1} psi>^{-1} = 1 since psi is normalized in the prior covariance norm. 
            shifted_sample = dl.Vector()
            self.prior.init_vector(shifted_sample, 1)
            shifted_sample.axpy(1.0, base_sample)
            shifted_sample.axpy((sigma - 1.0) * coeff, self.psi)

            # Compute C_i^{-1} m_shift
            shifted_precision = dl.Vector()
            self.prior.init_vector(shifted_precision, 1)
            shifted_prior.R.mult(shifted_sample, shifted_precision)

            self.shared_m_mc.append(shifted_sample)
            self.shared_Rm_mc.append(shifted_precision)

    def _prepare_component_solver(self, m_i):
        sigma_sq = float(self.mix_1d["sigma"] ** 2) # sigma for 1D GMM component, squared
        alpha = (sigma_sq - 1.0) * float(self.lambda_psi) # Coefficient for shifting
        prior_i = _ShiftedPrior(
            self.prior,
            m_i,
            self.psi,
            alpha,
            sigma_sq,
            self.lambda_psi,
        )

        # Use cost functional objective for a single Gaussian component
        solver = _TaylorQuadraticCVaRLegacy(
            self._component_settings(),
            self.model,
            prior_i,
            penalization=None,
            tol=self.tol,
        )

        # Replace the constructor-generated samples with covariance-consistent
        # mixture samples shared across components.
        solver.m_mc = self.shared_m_mc
        solver.Rm_mc = self.shared_Rm_mc

        solver._copy_z(self.z)
        solver._linearize_at_mean()
        solver._compute_eigendecomposition()
        solver._compute_mode_increments()
        solver._compute_surrogate_samples()
        return solver

    # Compute the cvar objective for the mixture
    def _mixture_cvar_objective(self, t):
        value = float(t)
        scale = 1.0 / (1.0 - self.beta)
        # Each component's contribution to cvar is 1/(1-beta) * w_i * np.mean(smoothplus(Q_surrogate - t)), w_i the component weight
        for i, solver in enumerate(self.component_solvers):
            value += self.component_weights[i] * np.mean(
                self.smoothplus(solver.Q_surrogate - t)
            ) * scale
        return float(value)

    # Compute the optimal t, which is the Value at Risk.
    def _compute_global_t(self):
        all_samples = np.concatenate([solver.Q_surrogate for solver in self.component_solvers])
        all_weights = np.concatenate(
            [
                np.full(solver.N_mc, self.component_weights[i] / solver.N_mc)
                for i, solver in enumerate(self.component_solvers)
            ]
        )
        t_init = _weighted_quantile(all_samples, all_weights, self.beta)

        def objective_for_fmin(t_arr):
            return self._mixture_cvar_objective(float(np.atleast_1d(t_arr)[0]))

        minimum = scipy.optimize.fmin(
            objective_for_fmin,
            t_init,
            disp=False,
            xtol=1e-10,
            ftol=1e-10,
        )
        t_opt = float(minimum[0])
        return t_opt, self._mixture_cvar_objective(t_opt)

    def _refresh_component_weights_and_adjoint_data(self):
        for i, solver in enumerate(self.component_solvers):
            solver.t_opt = self.t_opt
            solver.plus_grad = solver.smoothplus.grad(solver.Q_surrogate - self.t_opt) # The smoothplus derivative at sample k, E_k(t)
            # Keep the inner component adjoint solves identical to the
            # single-Gaussian quadratic CVaR normalization. The cluster weight
            # w_i is applied only when combining the final component gradients.
            scale = 1.0 / ((1.0 - self.beta) * solver.N_mc)
            solver.sample_weights = scale * solver.plus_grad
            solver.sample_weight_sum = float(np.sum(solver.sample_weights))
            solver._build_eigen_adjoint_vectors()
            solver._cache_objective_z()

    def objective(self):
        self._compute_direction()

        # Samples (m_i^{k} - \bar{m}_i)s for each cluster i, the samples are from N(0, C_i)
        sigma_sq = float(self.mix_1d["sigma"] ** 2)
        alpha = (sigma_sq - 1.0) * float(self.lambda_psi)
        shared_prior = _ShiftedPrior(
            self.prior,
            self.prior.mean,
            self.psi,
            alpha,
            sigma_sq,
            self.lambda_psi,
        )
        self._build_shared_component_samples(shared_prior)

        self.component_solvers = []
        self.component_Q0 = []
        self.component_d = []
        self.component_samples = []

        for i in range(self.N_mix):
            solver = self._prepare_component_solver(self.m_bar_i[i])
            self.component_solvers.append(solver)
            self.component_Q0.append(float(solver.Q_0))
            self.component_d.append(np.array(solver.d, copy=True))
            self.component_samples.append(np.array(solver.Q_surrogate, copy=True))

        # Compute the cvar objective
        self.t_opt, self.cvar = self._compute_global_t()
        self._refresh_component_weights_and_adjoint_data()
        self.Q_surrogate = np.concatenate(self.component_samples)

        if self.verbose and self.mpi_rank == 0:
            print(
                f"  [Mixture Quad CVaR] N_mix={self.N_mix}, "
                f"t_opt={self.t_opt:.4e}, CVaR={self.cvar:.4e}"
            )
            print(
                "                        component Q0[:min(5,N_mix)]={}".format(
                    self.component_Q0[: min(5, len(self.component_Q0))]
                )
            )

        return self.cvar

    # Compute the z-gradient contribution from eah component
    def _component_z_gradient(self, solver):
        """Evaluate one component's z-gradient with the already-fixed global t_opt."""
        solver.solveAdjIncrementalAdj()
        solver.solveAdjIncrementalFwd()
        solver.solveAdjAdj()
        solver.solveAdjFwd()

        dzq = solver.model.generate_vector(CONTROL)

        x_fun = vector2Function(solver.x, solver.pde.Vh[STATE])
        y_fun = vector2Function(solver.y, solver.pde.Vh[ADJOINT])
        m_fun = vector2Function(solver.m, solver.pde.Vh[PARAMETER])
        z_fun = vector2Function(solver.z, solver.pde.Vh[CONTROL])
        form = solver.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        z_test = dl.TestFunction(solver.pde.Vh[CONTROL])

        # Compute 1/(1-beta)K_i E_ik(t_opt) * <\tilde z, r_mz (m_i^k - \bar m_i)>
        for k in range(solver.N_mc):
            mhat_fun = vector2Function(solver.m_mc[k], solver.pde.Vh[PARAMETER])
            dmr = dl.derivative(form, m_fun, mhat_fun)
            dmzr = solver.model.generate_vector(CONTROL)
            dl.assemble(dl.derivative(dmr, z_fun, z_test), tensor=dmzr)
            dzq.axpy(solver.sample_weights[k], dmzr) # Sample weight here contains w_i

        ystar_fun = vector2Function(solver.ystar, solver.pde.Vh[ADJOINT])
        dyr = dl.derivative(form, y_fun, ystar_fun)
        dyzr = solver.model.generate_vector(CONTROL)
        dl.assemble(dl.derivative(dyr, z_fun, z_test), tensor=dyzr)
        dzq.axpy(1.0, dyzr)

        xstar_fun = vector2Function(solver.xstar, solver.pde.Vh[STATE])
        dxr = dl.derivative(form, x_fun, xstar_fun)
        dxzr = solver.model.generate_vector(CONTROL)
        dl.assemble(dl.derivative(dxr, z_fun, z_test), tensor=dxzr)
        dzq.axpy(1.0, dxzr)

        for i in range(solver.N_tr):
            (
                _dyzr,
                _dxzr,
                dyxzr,
                dymzr,
                dxyzr,
                dxxzr,
                dxmzr,
                dmmzr,
                dmyzr,
                dmxzr,
                dmxzq,
                dmmzq,
                dxxzq,
                dxmzq,
            ) = solver.pde.gradientControl(
                solver.x_all,
                solver.xstar,
                solver.ystar,
                solver.xhat[i],
                solver.xhatstar[i],
                solver.mhat[i],
                solver.mhatstar[i],
                solver.yhat[i],
                solver.yhatstar[i],
                solver.qoi,
            )

            dzq.axpy(1.0, dyxzr)
            dzq.axpy(1.0, dymzr)
            dzq.axpy(1.0, dxyzr)
            dzq.axpy(1.0, dxxzr)
            dzq.axpy(1.0, dxmzr)
            dzq.axpy(1.0, dmmzr)
            dzq.axpy(1.0, dmyzr)
            dzq.axpy(1.0, dmxzr)

            # The following four terms are zero if Q only explicitly depends on state x
            dzq.axpy(1.0, dmxzq)
            dzq.axpy(1.0, dmmzq)
            dzq.axpy(1.0, dxxzq)
            dzq.axpy(1.0, dxmzq)

        return dzq

    def costValue(self, z):
        self.func_ncalls += 1
        self._copy_z(z)

        objective = self.objective()
        self._cache_objective_z()
        self.grad_cache = None

        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalty

    def costGradient(self, z):
        self.grad_ncalls += 1
        self._copy_z(z)

        if not self._objective_matches_current_z():
            self.objective()
            self._cache_objective_z()

        dz = self.model.generate_vector(CONTROL)

        # Final mixture gradient is the weighted combination sum_i w_i * grad_i.
        for i, solver in enumerate(self.component_solvers):
            dz.axpy(self.component_weights[i], self._component_z_gradient(solver))

        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        self.grad_cache = dz.copy()
        return dz, np.sqrt(dz.inner(dz))


class TaylorMixtureQuadraticCVaRControlCostFunctional(ControlCostFunctional):
    """Gaussian mixture with quadratic Taylor CVaR approximation."""

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_mixture_quadratic_cvar_settings(settings)
        self._legacy = _TaylorMixtureQuadraticCVaRLegacy(
            self.settings,
            control_model,
            prior,
            penalization,
            tol,
        )

    @property
    def cvar(self):
        return self._legacy.cvar

    @property
    def t_opt(self):
        return self._legacy.t_opt

    @property
    def component_Q0(self):
        return self._legacy.component_Q0

    @property
    def component_d(self):
        return self._legacy.component_d

    @property
    def component_samples(self):
        return self._legacy.component_samples

    @property
    def component_weights(self):
        return self._legacy.component_weights

    @property
    def Q_surrogate(self):
        return self._legacy.Q_surrogate

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


__all__ = ["TaylorMixtureQuadraticCVaRControlCostFunctional"]
