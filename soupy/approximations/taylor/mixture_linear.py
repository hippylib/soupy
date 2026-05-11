"""Gaussian mixture with linear Taylor approximation for mean + variance.

The implementation follows the same z-gradient workflow used in linear.py:
- solve forward/adjoint at each linearization point
- build variance sensitivity through incremental x*/y* solves
- assemble control gradient from direct and incremental terms

For mixture objective
    J = E[Q_mix] + beta * Var[Q_mix]
with
    E[Q_mix] = sum_i w_i mu_i,
    Var[Q_mix] = sum_i w_i (mu_i^2 + var_i) - E[Q_mix]^2,
the gradient is assembled component-wise as a weighted linear combination.
"""

from __future__ import annotations

from typing import Optional, Union

import dolfin as dl
import numpy as np
from mpi4py import MPI
from hippylib import MultiVector, Random, vector2Function
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .gmm_library import get_1d_gmm_library_mixture
from .settings import taylor_mixture_linear_settings


def gaussian_mixture_mean_variance(means, stds, weights):
    """
    Compute mean and variance of a Gaussian mixture distribution:
        E[Q_mix] = sum_i w_i mu_i,
        Var[Q_mix] = sum_i w_i (mu_i^2 + var_i) - E[Q_mix]^2
    """
    means = np.asarray(means)
    stds = np.asarray(stds)
    weights = np.asarray(weights)

    mean = np.sum(weights * means)
    second_moment = np.sum(weights * (means ** 2 + stds ** 2))
    variance = second_moment - mean ** 2

    return mean, variance


class _TaylorMixtureLinearLegacy:
    """Implementation of Gaussian mixture linear Taylor mean + variance."""

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

        self.x = model.generate_vector(STATE)
        self.y = model.generate_vector(STATE)
        self.x_all = [self.x, prior.mean, self.y, self.z]

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.beta = settings["beta"]
        self.N_mix = settings["N_mix"]
        self.direction = settings["direction"]
        self.seed = settings["seed"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())

        self.mix_1d = get_1d_gmm_library_mixture(
            self.N_mix, rule=1, warn=(self.mpi_rank == 0)
        )
        self.component_weights = self.mix_1d["weights"]

        self.psi = None
        self.lambda_psi = None
        self.m_bar_i = []
        self.direction_computed = False

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        self.component_means = []
        self.component_stds = []
        self.component_vars = []

        self.x_components = []
        self.y_components = []
        self.Cdmq_components = []

        self.mixture_mean = 0.0
        self.mixture_var = 0.0

        self.grad_cache = None

    def _z_has_changed(self, z):
        """Return True when the incoming control differs from the cached one."""
        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            z_local = z[idx[0] : idx[1]]
            delta = self.z.get_local() - z_local
            return np.dot(delta, delta) > 1e-20

        self.z_diff.zero()
        self.z_diff.axpy(1.0, self.z)
        self.z_diff.axpy(-1.0, z)
        return self.z_diff.inner(self.z_diff) > 1e-20

    def _compute_direction(self):
        """Compute decomposition direction (KLE or HEP)."""
        if self.direction_computed:
            return

        self.x_all[CONTROL] = self.z
        self.x_all[PARAMETER] = self.prior.mean

        # Update PDE state given the control and mean parameter
        self.pde.solveFwd(self.x, self.x_all)
        self.x_all[STATE] = self.x

        rhs = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(self.x_all, rhs) # Compute the RHS for the adjoint equation with the new state. 
        self.pde.solveAdj(self.y, self.x_all, rhs) # Update the adjoint variable y in x_all with the new RHS.
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False)
        self.qoi.setLinearizationPoint(self.x_all)

        if self.direction == "hep":
            omega = MultiVector(self.pde.generate_parameter(), 15)
            rand = Random(seed=self.seed)
            for i in range(15):
                rand.normal(1.0, omega[i])

            d, U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, omega.nvec(), s=1)
            dominant_idx = int(np.argmax(np.abs(d)))
            dominant_eigenvalue = float(d[dominant_idx])

            self.psi = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.psi.zero()
            self.psi.axpy(1.0, U[dominant_idx])
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print(f"  [Mixture] Using HEP direction, dominant eigenvalue = {dominant_eigenvalue:.4e}")
        else:
            v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            rand = Random(seed=self.seed)
            rand.normal(1.0, v)

            # Power iteration for KLE direction. 
            for _ in range(50):
                w = dl.Function(self.pde.Vh[PARAMETER]).vector()
                # Computing w = R^{-1} v = Cv, setting the new v to W, and doing normalization. 
                # After many iterations, v should converge to the dominant eigenvector of C. 
                self.prior.Rsolver.solve(w, v)
                norm_w = np.sqrt(w.inner(w))
                if norm_w > 1e-14:
                    v.zero()
                    v.axpy(1.0 / norm_w, w)

            # Set self.psi to the dominant eigenvector divided by the sqrt of the corresponding eigenvlue
            R_v = dl.Function(self.pde.Vh[PARAMETER]).vector()
            self.prior.R.mult(v, R_v)
            norm_sq = v.inner(R_v)
            v_local = v.get_local()
            v.set_local(v_local / np.sqrt(norm_sq))
            v.apply("")

            self.psi = v
            # The self.psi is already the dominant eigenvector divided by the eigenvalue lambda, so that now weset self. lambda_psi
            # to 1.0 to ensure that sqrt{lambda}*psi = self.psi * sqrt{self.lambda} is the same. 
            self.lambda_psi = 1.0

            if self.verbose and self.mpi_rank == 0:
                print("  [Mixture] Using KLE direction")

        mix_means_1d = self.mix_1d["means"]
        sqrt_lambda = np.sqrt(self.lambda_psi)

        # Precompute the shifted means for each mixture component: m_bar_i = m_0 + mean_1d_i * sqrt(lambda) * psi, where m_0 is the prior mean.
        self.m_bar_i = []
        for i in range(self.N_mix):
            m_i = dl.Function(self.pde.Vh[PARAMETER]).vector()
            m_i.axpy(1.0, self.prior.mean)
            m_i.axpy(mix_means_1d[i] * sqrt_lambda, self.psi)
            self.m_bar_i.append(m_i)

        self.direction_computed = True

    def objective(self):
        """Compute mean + variance objective using Gaussian mixture linear Taylor."""
        self._compute_direction()

        sigma_sq = self.mix_1d["sigma"] ** 2
        alpha = (sigma_sq - 1.0) * self.lambda_psi

        self.component_means = []
        self.component_stds = []
        self.component_vars = []

        self.x_components = []
        self.y_components = []
        self.Cdmq_components = []

        for i in range(self.N_mix):
            m_i = self.m_bar_i[i]

            x_i = self.model.generate_vector(STATE)
            y_i = self.model.generate_vector(STATE)
            x_all_i = [x_i, m_i, y_i, self.z]

            self.pde.solveFwd(x_i, x_all_i)
            x_all_i[STATE] = x_i

            Q_i = self.qoi.cost(x_all_i)
            self.component_means.append(Q_i)

            rhs_i = self.model.generate_vector(STATE)
            self.qoi.adj_rhs(x_all_i, rhs_i)
            self.pde.solveAdj(y_i, x_all_i, rhs_i)
            x_all_i[ADJOINT] = y_i

            self.pde.setLinearizationPoint(x_all_i, False)
            self.qoi.setLinearizationPoint(x_all_i)

            x_fun = vector2Function(x_i, self.pde.Vh[STATE])
            y_fun = vector2Function(y_i, self.pde.Vh[ADJOINT])
            m_fun = vector2Function(m_i, self.pde.Vh[PARAMETER])
            z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])

            f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
            m_test = dl.TestFunction(self.pde.Vh[PARAMETER])
            # Here m_test is an arbitrary perturbation instead of a particular direction
            dmq = dl.assemble(dl.derivative(f_form, m_fun, m_test))

            Cdmq_i = self.model.generate_vector(PARAMETER)
            # Compute the first term of <dmq_i, C_i dmq_i>, which is <dmq_i, C dmq_i>
            self.prior.Rsolver.solve(Cdmq_i, dmq)

            # Compute the second term of <dmq_i, C_i dmq_i>, which is (sigma^2 - 1) * lambda_psi * <dmq_i, psi> <psi, dmq_i>
            dmq_dot_psi = dmq.inner(self.psi)
            # alpha = (sigma^2 - 1) * lambda_psi. 
            Cdmq_i.axpy(alpha * dmq_dot_psi, self.psi)

            # Variance for cluster i is given by var_i = <dmq_i, C_i dmq_i> = <dmq_i, C dmq_i> + (sigma^2 - 1) * lambda_psi * <dmq_i, psi> <psi, dmq_i>
            var_i = dmq.inner(Cdmq_i)
            self.component_vars.append(var_i)
            self.component_stds.append(np.sqrt(max(0.0, var_i)))

            self.x_components.append(x_i)
            self.y_components.append(y_i)
            self.Cdmq_components.append(Cdmq_i)

        self.mixture_mean, mixture_var = gaussian_mixture_mean_variance(
            self.component_means,
            self.component_stds,
            self.component_weights,
        )
        self.mixture_var = max(0.0, mixture_var)

        if self.verbose and self.mpi_rank == 0:
            print(
                f"  [Mixture Linear] N_mix={self.N_mix}, Mean={self.mixture_mean:.4e}, Var={self.mixture_var:.4e}"
            )

        return self.mixture_mean + self.beta * self.mixture_var

    def _component_weighted_gradient(self, x_i, y_i, m_i, Cdmq_i, gamma_i, eta_i):
        """Compute weighted component gradient using the same workflow as linear.py.

        Component contribution:
            gamma_i * dmu_i/dz + eta_i * dvar_i/dz
        where 
        mu_i = Q(x_i), 
        var_i = <dmq_i, C_i dmq_i>, 
        gamma_i = w_i * (1.0 + 2.0 * self.beta * (mu_i - self.mixture_mean)), 
        eta_i = self.beta * w_i

        Variance formula for GMM (using within-cluster and between-cluster decomposition):
        Var[Q_mix] = sum_i w_i * var_i + sum_i w_i * (mu_i - self.mixture_mean)^2
        """
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

        # Variance derivative building blocks.
        dmrC = dl.derivative(form, m_fun, Cdmq_fun) # <D_m Q_i, C_i D_m Q_i>

        # We want to solve u star and v start in the incremental state equation in order to compute the final gradient. 
        xstarrhs = self.pde.generate_state()
        if eta_i != 0.0:
            # The RHS of incremental state equation \partial L / \partial v_i: 
            # -2*beta < \tilde{v}_i, \partial_{vm}\bar{r}_i (C_i\partial_m \bar{r}_i)>
            dmyrC = dl.assemble(dl.derivative(dmrC, y_fun, y_test)) # The RHS of incremental state equation \partial L / \partial v_i
            [bc.apply(dmyrC) for bc in self.pde.bc0]
            xstarrhs.axpy(2.0 * eta_i, dmyrC)

        # Solve the incremental state equation for x_star, which is indeed w_i * u_star 
        xstar = self.pde.generate_state()
        self.pde.solveIncremental(xstar, -xstarrhs, False)

        # Mean and variance terms in y* equation (same structure as linear.py).
        ystarrhs = self.pde.generate_state()

        # Solve the incremental adjoint equation for y_star
        # The incremental adjoint equation \partial L / \partial u_i (check the note for detailed equation)
        
        #  <\tilde{u}_i, （1 + 2 * beta - 2 * beta * self.mixture_mean)\partial_{u} \bar{Q}_i>
        if gamma_i != 0.0:
            dxq = self.pde.generate_state()
            self.qoi.grad(STATE, [x_i, m_i, y_i, self.z], dxq)
            [bc.apply(dxq) for bc in self.pde.bc0]
            ystarrhs.axpy(gamma_i, dxq)

        # -2*beta <\tilde{u}_i, \partial_{um}\bar{r}_i (C_i\partial_m \bar{r}_i)>
        if eta_i != 0.0:
            dmxrC = dl.assemble(dl.derivative(dmrC, x_fun, x_test))
            [bc.apply(dmxrC) for bc in self.pde.bc0]
            ystarrhs.axpy(2.0 * eta_i, dmxrC)

        # - <\tilde{u}_i, \partial_{uu}\bar{r}_i u_istar>
        xstar_fun = vector2Function(xstar, Vh[STATE])
        dxr = dl.derivative(form, x_fun, xstar_fun)
        dxxr = dl.assemble(dl.derivative(dxr, x_fun, x_test))
        [bc.apply(dxxr) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxr)

        # - <\tilde{u}_i, \partial_{uu}\bar{Q}_i u_istar>
        dxxq = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, xstar, dxxq)
        [bc.apply(dxxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxq)

        ystar = self.pde.generate_state()
        self.pde.solveIncremental(ystar, -ystarrhs, True)

        # Now we have computed x, y, x_star, and y_star. We are well-prepared to compute the final gradient. 
        grad = self.model.generate_vector(CONTROL)

        # This is 2 * beta * w_i<\tilde{z}, \partial_{zm}\bar{r}_i (C_i\partial_m \bar{r}_i)>. 
        if eta_i != 0.0:
            dmzrC = dl.assemble(dl.derivative(dmrC, z_fun, z_test))
            grad.axpy(2.0 * eta_i, dmzrC)

        # This is w_i * <\tilde{z}, \partial_{zv} \bar{r}_i v_istar>.
        ystar_fun = vector2Function(ystar, Vh[ADJOINT])
        dyr = dl.derivative(form, y_fun, ystar_fun)
        dyzr = dl.assemble(dl.derivative(dyr, z_fun, z_test))
        grad.axpy(1.0, dyzr)

        # This is w_i * <\tilde{z}, \partial_{zu} \bar{r}_i u_istar>
        dxzr = dl.assemble(dl.derivative(dxr, z_fun, z_test))
        grad.axpy(1.0, dxzr)

        return grad

    def costValue(self, z, FD_gradient_check=False):
        """Evaluate cost at control z."""
        self.func_ncalls += 1
        z_changed = self._z_has_changed(z)

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0] : idx[1]])
            self.z.apply("")
        else:
            self.z.zero()
            self.z.axpy(1.0, z)

        if z_changed and not FD_gradient_check:
            self.direction_computed = False
        objective = self.objective()

        penalty = 0.0
        if self.penalization is not None:
            penalty = self.penalization.cost(self.z)

        self.grad_cache = None
        return objective + penalty

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

            gamma_i = w_i * (1.0 + 2.0 * self.beta * (mu_i - self.mixture_mean))
            eta_i = self.beta * w_i

            dcomp_i = self._component_weighted_gradient(
                x_i, y_i, m_i, Cdmq_i, gamma_i, eta_i
            )
            dz.axpy(1.0, dcomp_i)

        if self.penalization is not None:
            dzp = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, dzp)
            dz.axpy(1.0, dzp)

        self.grad_cache = dz.copy()
        return dz, np.sqrt(dz.inner(dz))


class TaylorMixtureLinearControlCostFunctional(ControlCostFunctional):
    """Gaussian mixture with linear Taylor mean + variance approximation."""

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_mixture_linear_settings(settings)
        self._legacy = _TaylorMixtureLinearLegacy(
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
    def component_means(self):
        return self._legacy.component_means

    @property
    def component_stds(self):
        return self._legacy.component_stds

    @property
    def component_weights(self):
        return self._legacy.component_weights

    def generate_vector(self, component="ALL"):
        return self._legacy.model.generate_vector(component)

    def cost(self, z, order=0, FD_gradient_check=False):
        value = self._legacy.costValue(z, FD_gradient_check=FD_gradient_check)
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


__all__ = ["TaylorMixtureLinearControlCostFunctional", "gaussian_mixture_mean_variance"]
