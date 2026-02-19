"""Gaussian mixture with quadratic Taylor approximation for mean + variance.

Implementation strategy:
- Build Gaussian-mixture component means in parameter space.
- For each component mean, solve a *single* quadratic Taylor subproblem using
  the validated implementation in ``quadratic.py``.
- Combine component moments and sensitivities through mixture weights.

This keeps the core numerical logic aligned with ``quadratic.py`` rather than
introducing a separate derivative pipeline.
"""

from __future__ import annotations

from typing import Optional, Union

import dolfin as dl
import numpy as np
from hippylib import MultiVector, Random
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .mixture_data import get_1d_mixture
from .quadratic import _TaylorQuadraticLegacy
from .settings import taylor_mixture_quadratic_settings, taylor_quadratic_settings


def quadratic_mixture_mean_variance_analytical(
    component_Q0, component_lin_var, component_d, weights, sigma_sq
):
    """Compute mean/variance of a quadratic-Taylor Gaussian mixture analytically.

    The helper signature is preserved for backward compatibility with existing
    imports. Internally, this follows the standard mixture moment formulas.
    """
    n_mix = len(weights)
    component_means = []
    component_vars = []

    for i in range(n_mix):
        Q0_i = component_Q0[i]
        lin_var_i = component_lin_var[i]
        d_i = np.asarray(component_d[i])

        mean_i = Q0_i + 0.5 * np.sum(d_i)
        var_i = lin_var_i + 0.5 * np.sum(d_i ** 2)

        component_means.append(mean_i)
        component_vars.append(var_i)

    component_means = np.asarray(component_means)
    component_vars = np.asarray(component_vars)
    weights = np.asarray(weights)

    mixture_mean = np.sum(weights * component_means)
    second_moment = np.sum(weights * (component_means ** 2 + component_vars))
    mixture_var = second_moment - mixture_mean ** 2

    return float(mixture_mean), float(mixture_var)


class _ShiftedPrior:
    """Prior view with shifted mean and shared covariance operators."""

    def __init__(self, base_prior, mean_vec):
        self._base = base_prior
        self.mean = mean_vec
        self.R = base_prior.R
        self.Rsolver = base_prior.Rsolver

    def __getattr__(self, name):
        return getattr(self._base, name)


class _TaylorMixtureQuadraticLegacy:
    """Gaussian-mixture quadratic Taylor risk objective and gradient."""

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

        self.x = model.generate_vector(STATE)
        self.y = model.generate_vector(STATE)
        self.x_all = [self.x, prior.mean, self.y, self.z]

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.beta = settings["beta"]
        self.N_mix = settings["N_mix"]
        self.direction = settings["direction"]
        self.N_tr = settings["N_tr"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())

        self.mix_1d = get_1d_mixture(self.N_mix)
        self.component_weights = np.asarray(self.mix_1d["weights"])

        self.psi = None
        self.lambda_psi = None
        self.m_bar_i = []
        self.direction_computed = False

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        self.component_Q0 = []
        self.component_lin_var = []
        self.component_d = []
        self.component_quad_mean = []
        self.component_quad_var = []

        self.component_dmu = []
        self.component_dvar = []

        self.mixture_mean = 0.0
        self.mixture_var = 0.0

        self.grad_cache = None

    def _copy_z(self, z):
        self.z.zero()
        self.z.axpy(1.0, z)

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

    def _compute_direction(self):
        """Compute decomposition direction (KLE or HEP)."""
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

            d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, 1, s=1)

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.axpy(1.0, U[0])
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print(f"  [Mixture Quad] Using HEP direction, dominant eigenvalue = {d[0]:.4e}")
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
                print("  [Mixture Quad] Using KLE direction")

        mix_means_1d = self.mix_1d["means"]
        sqrt_lambda = np.sqrt(self.lambda_psi)

        self.m_bar_i = []
        for i in range(self.N_mix):
            m_i = dl.Function(self.pde.Vh[PARAMETER]).vector()
            m_i.axpy(1.0, self.prior.mean)
            m_i.axpy(mix_means_1d[i] * sqrt_lambda, self.psi)
            self.m_bar_i.append(m_i)

        self.direction_computed = True

    def _component_settings(self, beta_val):
        return taylor_quadratic_settings(
            {
                "beta": beta_val,
                "N_tr": self.N_tr,
                "correction": False,
                "N_mc": 0,
                "verbose": False,
            }
        )

    def _evaluate_component(self, m_i):
        """Evaluate one component via the validated quadratic legacy solver.

        Returns:
            dict with mu, var, dmu, dvar and diagnostic component fields.
        """
        prior_i = _ShiftedPrior(self.prior, m_i)

        # beta=0 gives pure mean objective and mean gradient.
        legacy_mu = _TaylorQuadraticLegacy(
            self._component_settings(0.0),
            self.model,
            prior_i,
            penalization=None,
            tol=self.tol,
        )
        mu_i = legacy_mu.costValue(self.z)
        dmu_i, _ = legacy_mu.costGradient(self.z)

        # beta=1 gives mu + var - mu^2 and its gradient.
        legacy_beta1 = _TaylorQuadraticLegacy(
            self._component_settings(1.0),
            self.model,
            prior_i,
            penalization=None,
            tol=self.tol,
        )
        j_beta1 = legacy_beta1.costValue(self.z)
        g_beta1, _ = legacy_beta1.costGradient(self.z)

        # Recover var and dvar from algebraic identities.
        var_i = j_beta1 - mu_i + mu_i ** 2
        dvar_i = self.model.generate_vector(CONTROL)
        dvar_i.zero()
        dvar_i.axpy(1.0, g_beta1)
        dvar_i.axpy(-(1.0 - 2.0 * mu_i), dmu_i)

        Q0_i = legacy_beta1.Q_0
        # Linear variance term <g, Cg> extracted from legacy linear moment part.
        lin_var_i = legacy_beta1.lin_var - Q0_i ** 2
        d_i = np.array(legacy_beta1.d, copy=True)

        return {
            "mu": float(mu_i),
            "var": float(var_i),
            "dmu": dmu_i,
            "dvar": dvar_i,
            "Q0": float(Q0_i),
            "lin_var": float(lin_var_i),
            "d": d_i,
        }

    def objective(self):
        """Compute mixture objective and cache per-component sensitivities."""
        self._compute_direction()

        self.component_Q0 = []
        self.component_lin_var = []
        self.component_d = []
        self.component_quad_mean = []
        self.component_quad_var = []
        self.component_dmu = []
        self.component_dvar = []

        for i in range(self.N_mix):
            comp = self._evaluate_component(self.m_bar_i[i])
            self.component_Q0.append(comp["Q0"])
            self.component_lin_var.append(comp["lin_var"])
            self.component_d.append(comp["d"])
            self.component_quad_mean.append(comp["mu"])
            self.component_quad_var.append(comp["var"])
            self.component_dmu.append(comp["dmu"])
            self.component_dvar.append(comp["dvar"])

        mu = np.asarray(self.component_quad_mean)
        var = np.asarray(self.component_quad_var)
        w = self.component_weights

        self.mixture_mean = float(np.sum(w * mu))
        second_moment = np.sum(w * (mu ** 2 + var))
        self.mixture_var = float(second_moment - self.mixture_mean ** 2)

        if self.verbose and self.mpi_rank == 0:
            print(
                f"  [Mixture Quad] N_mix={self.N_mix}, Mean={self.mixture_mean:.4e}, Var={self.mixture_var:.4e}"
            )

        return self.mixture_mean + self.beta * self.mixture_var

    def costValue(self, z):
        """Evaluate cost at control z."""
        self.func_ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0] : idx[1]])
            self.z.apply("")
        else:
            self._copy_z(z)

        self.direction_computed = False
        objective = self.objective()
        self._cache_objective_z()

        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        self.grad_cache = None
        return objective + penalty

    def costGradient(self, z):
        """Compute mixture gradient via weighted component moments/sensitivities."""
        self.grad_ncalls += 1

        if isinstance(z, np.ndarray):
            self.costValue(z)
        else:
            self._copy_z(z)
            if not self._objective_matches_current_z():
                self.objective()
                self._cache_objective_z()

        dz = self.model.generate_vector(CONTROL)

        # dJ = sum_i w_i*(1 + 2*beta*(mu_i - mu_mix))*dmu_i + beta*sum_i w_i*dvar_i
        for i in range(self.N_mix):
            w_i = self.component_weights[i]
            mu_i = self.component_quad_mean[i]

            mean_coeff = w_i * (1.0 + 2.0 * self.beta * (mu_i - self.mixture_mean))
            var_coeff = self.beta * w_i

            dz.axpy(mean_coeff, self.component_dmu[i])
            dz.axpy(var_coeff, self.component_dvar[i])

        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        self.grad_cache = dz.copy()
        return dz, np.sqrt(dz.inner(dz))


class TaylorMixtureQuadraticControlCostFunctional(ControlCostFunctional):
    """Gaussian mixture with quadratic Taylor mean + variance approximation."""

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_mixture_quadratic_settings(settings)
        self._legacy = _TaylorMixtureQuadraticLegacy(
            self.settings,
            control_model,
            prior,
            penalization,
            tol,
        )

    @property
    def mixture_mean(self):
        return self._legacy.mixture_mean

    @property
    def mixture_var(self):
        return self._legacy.mixture_var

    @property
    def component_Q0(self):
        return self._legacy.component_Q0

    @property
    def component_quad_mean(self):
        return self._legacy.component_quad_mean

    @property
    def component_quad_var(self):
        return self._legacy.component_quad_var

    @property
    def component_d(self):
        return self._legacy.component_d

    @property
    def component_weights(self):
        return self._legacy.component_weights

    def generate_vector(self, component="ALL"):
        return self._legacy.model.generate_vector(component)

    def cost(self, z, order=0):
        value = self._legacy.costValue(z)
        self._legacy.grad_cache = None
        if order >= 1:
            dz, _ = self._legacy.costGradient(z)
            self._legacy.grad_cache = dz
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
    "TaylorMixtureQuadraticControlCostFunctional",
    "quadratic_mixture_mean_variance_analytical",
]
