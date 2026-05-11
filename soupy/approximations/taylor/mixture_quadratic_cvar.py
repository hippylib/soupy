"""Gaussian-mixture quadratic Taylor approximation for CVaR.

Implementation strategy:
- Build Gaussian-mixture component means in parameter space exactly as in
  ``mixture_quadratic.py``.
- For each component mean, run the validated single-Gaussian quadratic CVaR
  pipeline from ``quadratic_cvar.py`` up to the surrogate-sample stage.
- Evaluate the weighted mixture CVaR objective at the current augmented
  optimization variable ``(z, t)``.
- Reuse the full adjoint-based z-gradient machinery from
  ``quadratic_cvar.py`` component-by-component, with the only change being that
  each component uses the current optimization scalar ``t`` and the same unweighted
  single-component sample factors as ``quadratic_cvar.py``

      1 / ((1 - beta) K_i) * E_ik(t),

  while the final mixture z-gradient is assembled as the weighted combination
  ``sum_i w_i * grad_i``.

This follows the user-provided theory with an explicit augmented optimization
variable ``(z, t)``.
"""

from __future__ import annotations

from typing import Optional, Union

import dolfin as dl
import numpy as np
from hippylib import MultiVector, Random
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.augmentedVector import AugmentedVector
from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.smoothPlusApproximation import SmoothPlusApproximationQuartic
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .gmm_library import get_1d_gmm_library_mixture
from .mixture_quadratic import _ShiftedPrior
from .quadratic_cvar import _TaylorQuadraticCVaRLegacy
from .settings import taylor_mixture_quadratic_cvar_settings, taylor_quadratic_cvar_settings

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
        self.zt = AugmentedVector(self.z, copy_vector=False)
        self.z_at_objective = model.generate_vector(CONTROL)
        self.z_diff = model.generate_vector(CONTROL)
        self.t = 0.0
        self.t_at_objective = 0.0
        self.t_diff = 0.0
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
        self.seed = settings["seed"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        if self.N_mc <= 0:
            raise RuntimeError("mixture quadratic CVaR requires N_mc > 0.")

        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        self.mix_1d = get_1d_gmm_library_mixture(
            self.N_mix, rule=1, warn=(self.mpi_rank == 0)
        )
        self.component_weights = np.asarray(self.mix_1d["weights"], dtype=float)

        self.psi = None
        self.lambda_psi = None
        self.m_bar_i = []
        self.direction_computed = False
        self._direction_version = 0

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)
        self._hep_omega_template = MultiVector(self.pde.generate_parameter(), 11)
        hep_rand = Random(seed=self.seed)
        for i in range(11):
            hep_rand.normal(1.0, self._hep_omega_template[i])
        self._kle_v_template = self.model.generate_vector(PARAMETER)
        kle_rand = Random(seed=self.seed)
        kle_rand.normal(1.0, self._kle_v_template)
        self.smoothplus = SmoothPlusApproximationQuartic(epsilon=self.epsilon)

        self.component_solvers = []
        self.component_Q0 = []
        self.component_d = []
        self.component_samples = []
        self.shared_m_mc = []
        self.shared_Rm_mc = []
        self._shared_samples_direction_version = -1

        self.Q_surrogate = np.zeros(0)
        self.t_opt = 0.0
        self.cvar = 0.0

    def _copy_zt(self, zt):
        self.z.zero()
        if isinstance(zt, AugmentedVector):
            self.z.axpy(1.0, zt.get_vector())
            self.t = float(zt.get_scalar())
        elif isinstance(zt, np.ndarray):
            z_dim = self.z.local_range()[1] - self.z.local_range()[0]
            if zt.shape[0] == z_dim + 1:
                self.z.set_local(zt[:z_dim])
                self.z.apply("")
                self.t = float(zt[-1])
            else:
                idx = self.z.local_range()
                self.z.set_local(zt[idx[0] : idx[1]])
                self.z.apply("")
                self.t = 0.0
        elif hasattr(zt, "get_vector") and hasattr(zt, "get_scalar"):
            self.z.axpy(1.0, zt.get_vector())
            self.t = float(zt.get_scalar())
        else:
            self.z.axpy(1.0, zt)
            self.t = 0.0
        self.zt.set_scalar(self.t)
        self.objective_is_current = False

    def _cache_objective_zt(self):
        self.z_at_objective.zero()
        self.z_at_objective.axpy(1.0, self.z)
        self.t_at_objective = float(self.t)
        self.objective_is_current = True

    def _objective_matches_current_zt(self):
        if not self.objective_is_current:
            return False
        self.z_diff.zero()
        self.z_diff.axpy(1.0, self.z_at_objective)
        self.z_diff.axpy(-1.0, self.z)
        self.t_diff = self.t_at_objective - self.t
        return self.z_diff.inner(self.z_diff) <= 1e-20 and abs(self.t_diff) <= 1e-20

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
        """Compute the dominant mixture-splitting direction (KLE or HEP)."""

        # Comment the following lines for finite difference gradient test 
        if self.direction == "hep" and self.direction_computed and not FD_gradient_check:
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
            omega = MultiVector(self.pde.generate_parameter(), 11)
            for i in range(11):
                omega[i].zero()
                omega[i].axpy(1.0, self._hep_omega_template[i])

            # doublePassG returns eigenvectors U satisfying U^T R U = I, so
            # lambda_psi = <psi, R psi>^{-1} = 1 for the returned direction.
            d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, omega.nvec(), s=1)
            dominant_idx = int(np.argmax(np.abs(d)))
            dominant_eigenvalue = float(d[dominant_idx])

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.zero()
            self.psi.axpy(1.0, U[dominant_idx])
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print(
                    f"  [Mixture Quad CVaR] Using HEP direction, "
                    f"dominant eigenvalue = {dominant_eigenvalue:.4e}"
                )
        else:
            v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            v.zero()
            v.axpy(1.0, self._kle_v_template)

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
        self._direction_version += 1
        self._cache_direction_z()

    def _component_settings(self):
        return taylor_quadratic_cvar_settings(
            {
                "beta": self.beta,
                "N_tr": self.N_tr,
                "N_mc": self.N_mc,
                "seed": self.seed,
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
        if self._shared_samples_direction_version == self._direction_version:
            return

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

        self._shared_samples_direction_version = self._direction_version

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
        solver._mc_samples_initialized = True

        solver._copy_zt(self.zt)
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

    def _refresh_component_weights_and_adjoint_data(self):
        for i, solver in enumerate(self.component_solvers):
            solver.t = float(self.t)
            solver.t_opt = float(self.t)
            solver.cvar = float(
                self.t + np.mean(solver.smoothplus(solver.Q_surrogate - self.t)) / (1.0 - self.beta)
            )
            solver.plus_grad = solver.smoothplus.grad(solver.Q_surrogate - self.t) # The smoothplus derivative at sample k, E_k(t)
            # Keep the inner component adjoint solves identical to the
            # single-Gaussian quadratic CVaR normalization. The cluster weight
            # w_i is applied only when combining the final component gradients.
            scale = 1.0 / ((1.0 - self.beta) * solver.N_mc)
            solver.sample_weights = scale * solver.plus_grad
            solver.sample_weight_sum = float(np.sum(solver.sample_weights))
            solver._build_eigen_adjoint_vectors()
            solver._cache_objective_zt()

    def objective(self, FD_gradient_check=False):
        self._compute_direction(FD_gradient_check=FD_gradient_check)

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

        # Evaluate the mixture CVaR objective at the current optimization scalar t.
        self.t_opt = float(self.t)
        self.cvar = self._mixture_cvar_objective(self.t)
        self._refresh_component_weights_and_adjoint_data()
        self.Q_surrogate = np.concatenate(self.component_samples)

        if self.verbose and self.mpi_rank == 0:
            print(
                f"  [Mixture Quad CVaR] N_mix={self.N_mix}, "
                f"t={self.t:.4e}, CVaR={self.cvar:.4e}"
            )
            print(
                "                        component Q0[:min(5,N_mix)]={}".format(
                    self.component_Q0[: min(5, len(self.component_Q0))]
                )
            )

        return self.cvar

    def _component_z_gradient(self, solver):
        """Evaluate one component z-gradient by reusing the validated single-Gaussian solver."""
        component_grad, _ = solver.costGradient(self.zt)
        dzq = solver.model.generate_vector(CONTROL)
        dzq.zero()
        dzq.axpy(1.0, component_grad.get_vector())
        return dzq

    def costValue(self, zt, FD_gradient_check=False):
        self.func_ncalls += 1
        self._copy_zt(zt)

        objective = self.objective(FD_gradient_check=FD_gradient_check)
        self._cache_objective_zt()
        self.grad_cache = None

        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalty

    def costGradient(self, zt):
        self.grad_ncalls += 1
        self._copy_zt(zt)

        if not self._objective_matches_current_zt():
            self.objective()
            self._cache_objective_zt()

        dz = AugmentedVector(self.model.generate_vector(CONTROL), copy_vector=False)

        # Final mixture gradient is the weighted combination sum_i w_i * grad_i.
        for i, solver in enumerate(self.component_solvers):
            dz.get_vector().axpy(self.component_weights[i], self._component_z_gradient(solver))

        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.get_vector().axpy(1.0, pen)

        mixture_t_grad = 1.0 - sum(
            self.component_weights[i] * solver.sample_weight_sum
            for i, solver in enumerate(self.component_solvers)
        )
        dz.set_scalar(float(mixture_t_grad))

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
        if component == CONTROL:
            return AugmentedVector(self._legacy.model.generate_vector(CONTROL), copy_vector=False)
        return self._legacy.model.generate_vector(component)

    def cost(self, zt, order=0, FD_gradient_check=False):
        value = self._legacy.costValue(zt, FD_gradient_check=FD_gradient_check)
        if order >= 1:
            self._legacy.costGradient(zt)
        return value

    def grad(self, g):
        if self._legacy.grad_cache is None:
            dz, _ = self._legacy.costGradient(self._legacy.zt)
        else:
            dz = self._legacy.grad_cache
        g.zero()
        g.axpy(1.0, dz)
        self._legacy.grad_cache = None
        return np.sqrt(g.inner(g))


__all__ = ["TaylorMixtureQuadraticCVaRControlCostFunctional"]
