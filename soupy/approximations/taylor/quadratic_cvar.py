"""Second-order Taylor CVaR approximation based on a truncated Hessian spectrum.

The implementation follows the smooth CVaR surrogate

    J_quad(z, t)
      = t + 1 / ((1 - beta) K) sum_k [Q_quad^(k) - t]^+_eps + P(z),

where

    Q_quad^(k)
      = Q(bar m)
        + <Q_m(bar m), m^(k) - bar m>_M
        + 0.5 sum_j lambda_j
            <C^{-1} psi_j, m^(k) - bar m>_M^2.

The dominant eigenpairs (lambda_j, psi_j) are computed with the randomized
double-pass eigensolver applied to the reduced Hessian. The CVaR scalar
minimization over ``t`` uses the quartic smooth-plus approximation already
available in SOUPy.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import dolfin as dl
import numpy as np
import scipy.optimize
from hippylib import MultiVector, Random, vector2Function
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.augmentedVector import AugmentedVector
from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.smoothPlusApproximation import SmoothPlusApproximationQuartic
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .settings import taylor_quadratic_cvar_settings


def surrogate_cvar_from_samples(samples, beta, epsilon=1e-4):
    """Compute the smoothed CVaR from scalar samples by minimizing over ``t``."""
    samples = np.asarray(samples, dtype=float)
    quantile = float(np.percentile(samples, beta * 100.0))
    smoothplus = SmoothPlusApproximationQuartic(epsilon=epsilon)

    def cvar_obj(t):
        return float(t + np.mean(smoothplus(samples - t)) / (1.0 - beta))

    minimum = scipy.optimize.fmin(cvar_obj, quantile, disp=False, xtol=1e-10, ftol=1e-10) # Use the quantile as an initial guess
    # The quantile here is not the optimal solution as as the smooth-plus approximation is not exact, but it is a good initial guess. 
    t_opt = float(minimum[0])
    return t_opt, cvar_obj(t_opt)


class _TaylorQuadraticCVaRLegacy:
    """Quadratic Taylor CVaR approximation with adjoint-based z-gradient."""

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

        self.m = prior.mean
        self.x = model.generate_vector(STATE)
        self.y = model.generate_vector(STATE)
        self.x_all = [self.x, self.m, self.y, self.z]

        self.z_dir = model.generate_vector(CONTROL)
        self.grad_cache = None

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(STATE)
        self.rhs_adj2 = model.generate_vector(STATE)
        self.rhs_adj3 = model.generate_vector(STATE)
        self.rhs_adj4 = model.generate_vector(STATE)

        self.xstar = model.generate_vector(STATE)
        self.ystar = model.generate_vector(STATE)
        self.zero_state = model.generate_vector(STATE)
        self.mhelp = model.generate_vector(PARAMETER)
        self.Hmhat1 = model.generate_vector(PARAMETER)
        self.Cdmq = model.generate_vector(PARAMETER)
        self.dmq = model.generate_vector(PARAMETER)

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0

        self.beta = settings["beta"]
        self.N_tr = settings["N_tr"]
        self.N_mc = settings["N_mc"]
        self.seed = settings["seed"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        # The theory provided by the user does not include the old PDE correction.
        # Keep the setting for API compatibility but do not apply an extra correction.
        self.correction = False
        self.N_mc_correction = 0

        self.xhat = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.yhat = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.xhatstar = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.yhatstar = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.mhat = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]
        self.mhatstar = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)
        self._omega_template = MultiVector(self.pde.generate_parameter(), self.N_tr)
        rand = Random(seed=self.seed)
        for i in range(self.N_tr):
            rand.normal(1.0, self._omega_template[i])

        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        self.smoothplus = SmoothPlusApproximationQuartic(epsilon=self.epsilon)

        self.m_mc = []
        self.Rm_mc = []
        self._mc_samples_initialized = False

        self.d = np.zeros(self.N_tr)
        self.U = None
        self.Q_0 = 0.0
        self.t_opt = 0.0
        self.cvar = 0.0
        self.Q_surrogate = np.zeros(self.N_mc)
        self.linear_terms = np.zeros(self.N_mc)
        self.projections = np.zeros((self.N_tr, self.N_mc))
        self.plus_grad = np.zeros(self.N_mc)
        self.sample_weights = np.zeros(self.N_mc)
        self.sample_weight_sum = 0.0
        self.eig_adjoint_coeffs = np.zeros((self.N_tr, self.N_tr))

        self.tobj = 0.0
        self.tgrad = 0.0

    def _initialize_mc_samples(self):
        if self._mc_samples_initialized:
            return

        self.m_mc = []
        self.Rm_mc = []
        random_gen = Random(myid=0, nproc=self.mpi_size)
        for _ in range(self.N_mc):
            noise = dl.Vector()
            self.prior.init_vector(noise, "noise")
            random_gen.normal(1.0, noise)

            # Sample parameters m^(k) from the prior and store m^(k) - bar m
            sample = dl.Vector()
            self.prior.init_vector(sample, 1)
            self.prior.sample(noise, sample, add_mean=False)
            self.m_mc.append(sample)

            # Compute C^{-1} (m^(k) - bar m) for the sample
            rsample = dl.Vector()
            self.prior.init_vector(rsample, 1)
            self.prior.R.mult(sample, rsample)
            self.Rm_mc.append(rsample)

        self._mc_samples_initialized = True

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
                self.z.set_local(zt[idx[0]:idx[1]])
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

    def _linearize_at_mean(self):
        # Solve state equation for the state. 
        self.x_all[CONTROL] = self.z
        self.pde.solveFwd(self.x, self.x_all)
        self.Q_0 = self.qoi.cost(self.x_all)
        self.x_all[STATE] = self.x

        # Solve adjoint equation for the adjoint. 
        rhs = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(self.x_all, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs)
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False)
        self.qoi.setLinearizationPoint(self.x_all)

        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        # Compute \bar Q_m, which is  \bar r_m when the state and adjoint equations holds.  
        m_test = dl.TestFunction(self.pde.Vh[PARAMETER])
        self.dmq.zero()
        self.dmq.axpy(1.0, dl.assemble(dl.derivative(form, m_fun, m_test)))
        self.prior.Rsolver.solve(self.Cdmq, self.dmq)

    # Estimate the dominant eigenvalues and dominant eigenvectors (incremental states and adjoints are solved in the Hessian action)
    def _compute_eigendecomposition(self):
        omega = MultiVector(self.pde.generate_parameter(), self.N_tr)
        for i in range(self.N_tr):
            omega[i].zero()
            omega[i].axpy(1.0, self._omega_template[i])
        self.d, self.U = doublePassG(
            self.H, self.prior.R, self.prior.Rsolver, omega, self.N_tr, s=2
        )

    def _hessian_inner(self, mhat1, mhat2):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()

        # Solve incremental state for \hat u_j
        self.pde.apply_ij(ADJOINT, PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False)

        # Solve incremental adjoint for \hat v_j
        self.rhs_adj.zero()
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1.0, self.rhs_adj2)
        self.qoi.apply_ij(STATE, STATE, xhat, self.rhs_adj3)
        self.rhs_adj.axpy(1.0, self.rhs_adj3)
        self.qoi.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj4) # this term is zero is Q does not explicitly depends on m
        self.rhs_adj.axpy(1.0, self.rhs_adj4)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True)

        # Replace Q_mm \psi_j by r_{mm} \psi_j + r_{mu} \hat{u_j} + r_{mv}\hat{v_j}
        self.pde.apply_ij(PARAMETER, PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.pde.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.qoi.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp) # this term is zero is Q does not explicitly depends on m
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.qoi.apply_ij(PARAMETER, STATE, xhat, self.mhelp) # this term is zero is Q does not explicitly depends on m
        self.Hmhat1.axpy(1.0, self.mhelp)

        return mhat2.inner(self.Hmhat1), xhat, yhat

    def _compute_mode_increments(self):
        for i in range(self.N_tr):
            self.mhat[i].zero()
            self.mhat[i].axpy(1.0, self.U[i]) # Extracts each eigenvector psi_j
            _, xhat_i, yhat_i = self._hessian_inner(self.mhat[i], self.mhat[i]) # Extracts incremental states \hat u_j and incremental adjoint \hat v_j
            self.xhat[i].zero()
            self.xhat[i].axpy(1.0, xhat_i)
            self.yhat[i].zero()
            self.yhat[i].axpy(1.0, yhat_i)

    def _compute_surrogate_samples(self):
        self._initialize_mc_samples()
        self.Q_surrogate.fill(self.Q_0) # \bar Q
        for k in range(self.N_mc):
            lin_k = self.dmq.inner(self.m_mc[k]) # <Q_m(bar m), m^(k) - bar m>_M
            self.linear_terms[k] = lin_k
            self.Q_surrogate[k] += lin_k

        for j in range(self.N_tr):
            for k in range(self.N_mc):
                self.projections[j, k] = self.U[j].inner(self.Rm_mc[k]) # <C^{-1} psi_j, m^(k) - bar m>_M
            self.Q_surrogate += 0.5 * self.d[j] * self.projections[j, :] ** 2 # 0.5 sum_j lambda_j <C^{-1} psi_j, m^(k) - bar m>_M^2

        self.t_opt = float(self.t)
        self.cvar = float(
            self.t + np.mean(self.smoothplus(self.Q_surrogate - self.t)) / (1.0 - self.beta)
        )
        self.plus_grad = self.smoothplus.grad(self.Q_surrogate - self.t) # The gradient of smooth approx (E_1, ..., E_k)
        scale = 1.0 / ((1.0 - self.beta) * self.N_mc)
        self.sample_weights = scale * self.plus_grad
        self.sample_weight_sum = float(np.sum(self.sample_weights))

    def _build_eigen_adjoint_vectors(self):
        self.eig_adjoint_coeffs.fill(0.0)
        eig_tol = 1e-12

        # Solve the linear system: system*a = rhs for augmented coefficient vector a. 
        # The unknown vector to solve for is a = (a_j1, ..., a_jN_tr, lambda_j^*)^T, where psi_j^* \approx \sum_ell a_jell psi_ell
        # Solve for the coefficients a_jell = <psi_j^*, C^{-1} psi_ell> and lambda_j^*
        for j in range(self.N_tr):
            # Solve the projected system formed by:
            #   1) dL / d lambda_j = 0
            #   2) dL / d psi_j = 0, tested against the reduced basis {psi_ell}
            #
            # Unknowns are the reduced coefficients of psi_j^* in the basis
            # {psi_ell} together with the scalar lambda_j^*.
            system = np.zeros((self.N_tr + 1, self.N_tr + 1))
            rhs = np.zeros(self.N_tr + 1)

            # Projected psi_j-stationarity equations.
            for ell in range(self.N_tr):
                cross = float(
                    np.dot(
                        self.sample_weights, # Sample weights is 1/((1-beta)K) * E_k(t)
                        self.projections[j, :] * self.projections[ell, :],# The term 1/((1-beta)K) * sum_k p_{jk}^2, where p_{jk} = <C^{-1} psi_j, m^(k) - bar m>_M <C^{-1} psi_ell, m^(k) - bar m>_M 
                    )
                )

                # ell == j is the only time such that the term 2\lambda_j^* <psi_ell, C^{-1} psi_j> is 2*lambda_j but not zero
                if ell == j:
                    system[ell, -1] = 2.0 # The coefficient for lambda_j^*, which is a component of the unknown vector a
                    rhs[ell] = -self.d[j] * cross # The term 1/((1-beta)K) * lambda_j * sum_k p_{jk}^2, where p_{jk} = <C^{-1} psi_j, m^(k) - bar m>_M <C^{-1} psi_ell, m^(k) - bar m>_M 
                    # In this case the term <\psi_j^*, Q_mm psi_ell - \lambda_jC^{-1} psi_ell> is zero because Q_mm psi_ell = \lambda_jC^{-1} psi_ell  
                    continue

                gap = self.d[ell] - self.d[j]
                if abs(gap) <= eig_tol:
                    # In a numerically degenerate eigenspace we impose a zero
                    # off-diagonal coefficient as a gauge condition.
                    system[ell, ell] = 1.0
                    rhs[ell] = 0.0
                else:
                    # Here is the coefficient for psi_j^*, which is Q_mm psi_ell - lambda_j C^{-1} psi_ell
                    # r_mm psi_ell - lambda_j C^{-1} psi_ell = (lambda_ell - lambda_j) C^{-1} psi_ell
                    # So the term is actually (lambda_ell - lambda_j) <psi_j^*, C^{-1} psi_ell>
                    # Now we decompose psi_j^* = \sum_ell a_jell psi_ell, so the term is (lambda_ell - lambda_j) a_jell
                    system[ell, ell] = gap
                    rhs[ell] = -self.d[j] * cross # The term 1/((1-beta)K) * lambda_j * sum_k p_{jk}^2, where p_{jk} = <C^{-1} psi_j, m^(k) - bar m>_M <C^{-1} psi_ell, m^(k) - bar m>_M 

            # Projected lambda_j-stationarity equation:
            # <psi_j^*, C^{-1} psi_j> = 0.5 * sum_k w_k p_{jk}^2.
            system[-1, j] = 1.0
            rhs[-1] = 0.5 * float(
                np.dot(self.sample_weights, self.projections[j, :] ** 2)
            )

            # Solve linear system
            try:
                solution = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                solution, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)

            self.eig_adjoint_coeffs[j, :] = solution[: self.N_tr]

            # Use \sum_ell <\psi_j^*, C^{-1} \psi_ell> \psi_ell to approximate
            # the eigenvector adjoint \psi_j^* inside the truncated eigenspace.
            self.mhatstar[j].zero()
            for ell in range(self.N_tr):
                coeff = self.eig_adjoint_coeffs[j, ell]
                if abs(coeff) > 0.0:
                    self.mhatstar[j].axpy(coeff, self.U[ell])

    def objective(self):
        self._linearize_at_mean()
        self._compute_eigendecomposition()
        self._compute_mode_increments()
        self._compute_surrogate_samples()
        self._build_eigen_adjoint_vectors()

        if self.verbose and self.mpi_rank == 0:
            print(
                "  [Quadratic Taylor CVaR] Q0={:.3e}, CVaR={:.3e}, t={:.3e}".format(
                    self.Q_0, self.cvar, self.t
                )
            )
            print(
                "                          sum(E_k)/((1-beta)K)={:.6e}".format(
                    self.sample_weight_sum
                )
            )
            print(
                "                          eigenvalues[:5]={}".format(
                    self.d[: min(5, len(self.d))]
                )
            )

        return self.cvar

    def _forSolveAdjAdj(self, xhat, xhatstar, mhat, mhatstar):
        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        xhat_fun = vector2Function(xhat, self.pde.Vh[STATE])
        xhatstar_fun = vector2Function(xhatstar, self.pde.Vh[STATE])
        mhat_fun = vector2Function(mhat, self.pde.Vh[PARAMETER])
        mhatstar_fun = vector2Function(mhatstar, self.pde.Vh[PARAMETER])

        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        y_test = dl.TestFunction(self.pde.Vh[ADJOINT])

        dxr = dl.derivative(form, x_fun, xhatstar_fun)

        dxxr = dl.derivative(dxr, x_fun, xhat_fun)
        dxxyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dxxr, y_fun, y_test), tensor=dxxyr)
        [bc.apply(dxxyr) for bc in self.pde.bc0]

        dxmr = dl.derivative(dxr, m_fun, mhat_fun)
        dxmyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dxmr, y_fun, y_test), tensor=dxmyr)
        [bc.apply(dxmyr) for bc in self.pde.bc0]

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmr, y_fun, y_test), tensor=dmyr)
        [bc.apply(dmyr) for bc in self.pde.bc0]

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmmr = dl.derivative(dmr, m_fun, mhat_fun)
        dmmyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmmr, y_fun, y_test), tensor=dmmyr)
        [bc.apply(dmmyr) for bc in self.pde.bc0]

        dmxr = dl.derivative(dmr, x_fun, xhat_fun)
        dmxyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmxr, y_fun, y_test), tensor=dmxyr)
        [bc.apply(dmxyr) for bc in self.pde.bc0]

        return dmyr, dxxyr, dxmyr, dmmyr, dmxyr

    def _forSolveAdjFwd(self, xhat, xhatstar, mhat, mhatstar, yhat, yhatstar):
        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        xhat_fun = vector2Function(xhat, self.pde.Vh[STATE])
        xhatstar_fun = vector2Function(xhatstar, self.pde.Vh[STATE])
        mhat_fun = vector2Function(mhat, self.pde.Vh[PARAMETER])
        mhatstar_fun = vector2Function(mhatstar, self.pde.Vh[PARAMETER])
        yhat_fun = vector2Function(yhat, self.pde.Vh[ADJOINT])
        yhatstar_fun = vector2Function(yhatstar, self.pde.Vh[ADJOINT])

        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        x_test = dl.TestFunction(self.pde.Vh[STATE])

        dyr = dl.derivative(form, y_fun, yhatstar_fun)

        dyxr = dl.derivative(dyr, x_fun, xhat_fun)
        dyxxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dyxr, x_fun, x_test), tensor=dyxxr)
        [bc.apply(dyxxr) for bc in self.pde.bc0]

        dymr = dl.derivative(dyr, m_fun, mhat_fun)
        dymxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dymr, x_fun, x_test), tensor=dymxr)
        [bc.apply(dymxr) for bc in self.pde.bc0]

        dxr = dl.derivative(form, x_fun, xhatstar_fun)

        dxyr = dl.derivative(dxr, y_fun, yhat_fun)
        dxyxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dxyr, x_fun, x_test), tensor=dxyxr)
        [bc.apply(dxyxr) for bc in self.pde.bc0]

        dxxr = dl.derivative(dxr, x_fun, xhat_fun)
        dxxxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dxxr, x_fun, x_test), tensor=dxxxr)
        [bc.apply(dxxxr) for bc in self.pde.bc0]

        dxmr = dl.derivative(dxr, m_fun, mhat_fun)
        dxmxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dxmr, x_fun, x_test), tensor=dxmxr)
        [bc.apply(dxmxr) for bc in self.pde.bc0]

        dxxxq = self.pde.generate_state()
        self.qoi.apply_ijk(STATE, STATE, STATE, xhatstar, xhat, dxxxq)
        [bc.apply(dxxxq) for bc in self.pde.bc0]

        dmr = dl.derivative(form, m_fun, mhat_fun)
        dmxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmr, x_fun, x_test), tensor=dmxr)
        [bc.apply(dmxr) for bc in self.pde.bc0]

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmmr = dl.derivative(dmr, m_fun, mhat_fun)
        dmmxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmmr, x_fun, x_test), tensor=dmmxr)
        [bc.apply(dmmxr) for bc in self.pde.bc0]

        dmyr = dl.derivative(dmr, y_fun, yhat_fun)
        dmyxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmyr, x_fun, x_test), tensor=dmyxr)
        [bc.apply(dmyxr) for bc in self.pde.bc0]

        dmxr2 = dl.derivative(dmr, x_fun, xhat_fun)
        dmxxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmxr2, x_fun, x_test), tensor=dmxxr)
        [bc.apply(dmxxr) for bc in self.pde.bc0]

        dxmxq = self.pde.generate_state()
        self.qoi.apply_ijk(STATE, PARAMETER, STATE, xhatstar, mhat, dxmxq)
        [bc.apply(dxmxq) for bc in self.pde.bc0]

        dmmxq = self.pde.generate_state()
        self.qoi.apply_ijk(PARAMETER, PARAMETER, STATE, mhatstar, mhat, dmmxq)
        [bc.apply(dmmxq) for bc in self.pde.bc0]

        dmxxq = self.pde.generate_state()
        self.qoi.apply_ijk(PARAMETER, STATE, STATE, mhatstar, xhat, dmxxq)
        [bc.apply(dmxxq) for bc in self.pde.bc0]

        return (
            dmxr,
            dyxxr,
            dymxr,
            dxyxr,
            dxxxr,
            dxmxr,
            dxxxq,
            dmmxr,
            dmyxr,
            dmxxr,
            dxmxq,
            dmmxq,
            dmxxq,
        )

    def solveAdjIncrementalAdj(self):
        for i in range(self.N_tr):
            dmyr = self.pde.forSolveAdjIncrementalAdj(self.x_all, self.mhatstar[i])
            rhs = self.pde.generate_state()
            rhs.axpy(1.0, dmyr)
            self.xhatstar[i].zero()
            self.pde.solveIncremental(self.xhatstar[i], -rhs, False)

    def solveAdjIncrementalFwd(self):
        for i in range(self.N_tr):
            dmxr, dxxr, dxxq = self.pde.forSolveAdjIncrementalFwd(
                self.x_all, self.mhatstar[i], self.xhatstar[i], self.qoi
            )
            rhs = self.pde.generate_state()
            rhs.axpy(1.0, dmxr)
            rhs.axpy(1.0, dxxr)
            rhs.axpy(1.0, dxxq)
            self.yhatstar[i].zero()
            self.pde.solveIncremental(self.yhatstar[i], -rhs, True)

    def solveAdjAdj(self):
        xstarrhs = self.pde.generate_state()
        for k in range(self.N_mc):
            dmyr_k = self.pde.forSolveAdjIncrementalAdj(self.x_all, self.m_mc[k])
            xstarrhs.axpy(self.sample_weights[k], dmyr_k)

        for i in range(self.N_tr):
            _, dxxyr, dxmyr, dmmyr, dmxyr = self._forSolveAdjAdj(
                self.xhat[i], self.xhatstar[i], self.mhat[i], self.mhatstar[i]
            )
            xstarrhs.axpy(1.0, dxxyr)
            xstarrhs.axpy(1.0, dxmyr)
            xstarrhs.axpy(1.0, dmmyr)
            xstarrhs.axpy(1.0, dmxyr)

        self.xstar.zero()
        self.pde.solveIncremental(self.xstar, -xstarrhs, False)

    def solveAdjFwd(self):
        ystarrhs = self.pde.generate_state()
        qoi_grad = self.pde.generate_state()
        self.qoi.grad(STATE, self.x_all, qoi_grad)
        [bc.apply(qoi_grad) for bc in self.pde.bc0]
        ystarrhs.axpy(self.sample_weight_sum, qoi_grad)

        sample_zero = self.zero_state
        sample_zero.zero()
        for k in range(self.N_mc):
            dmxr_k, _, _ = self.pde.forSolveAdjIncrementalFwd(
                self.x_all, self.m_mc[k], sample_zero, self.qoi
            )
            ystarrhs.axpy(self.sample_weights[k], dmxr_k)

        ystarrhspde = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1.0, ystarrhspde)
        self.pde.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1.0, ystarrhspde)
        [bc.apply(ystarrhs) for bc in self.pde.bc0]

        for i in range(self.N_tr):
            (
                _dmxr,
                dyxxr,
                dymxr,
                dxyxr,
                dxxxr,
                dxmxr,
                dxxxq,
                dmmxr,
                dmyxr,
                dmxxr,
                _dxmxq,
                _dmmxq,
                _dmxxq,
            ) = self._forSolveAdjFwd(
                self.xhat[i],
                self.xhatstar[i],
                self.mhat[i],
                self.mhatstar[i],
                self.yhat[i],
                self.yhatstar[i],
            )
            ystarrhs.axpy(1.0, dyxxr)
            ystarrhs.axpy(1.0, dymxr)
            ystarrhs.axpy(1.0, dxyxr)
            ystarrhs.axpy(1.0, dxxxr)
            ystarrhs.axpy(1.0, dxmxr)
            ystarrhs.axpy(1.0, dxxxq)
            ystarrhs.axpy(1.0, dmmxr)
            ystarrhs.axpy(1.0, dmxxr)
            ystarrhs.axpy(1.0, dmyxr)

        self.ystar.zero()
        self.pde.solveIncremental(self.ystar, -ystarrhs, True)

    def costValue(self, zt):
        self.func_ncalls += 1
        self._copy_zt(zt)

        tobj = time.time()
        objective = self.objective()
        self.tobj = time.time() - tobj

        self._cache_objective_zt()
        self.grad_cache = None

        penalization = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalization

    def costGradient(self, zt):
        self.grad_ncalls += 1
        self._copy_zt(zt)

        if not self._objective_matches_current_zt():
            self.objective()
            self._cache_objective_zt()

        tgrad = time.time()
        self.solveAdjIncrementalAdj()
        self.solveAdjIncrementalFwd()
        self.solveAdjAdj()
        self.solveAdjFwd()

        dzq = self.model.generate_vector(CONTROL)

        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        z_test = dl.TestFunction(self.pde.Vh[CONTROL])

        # Final z-gradient closed-form expression. 
        for k in range(self.N_mc):
            mhat_fun = vector2Function(self.m_mc[k], self.pde.Vh[PARAMETER])
            dmr = dl.derivative(form, m_fun, mhat_fun)
            dmzr = self.model.generate_vector(CONTROL)
            dl.assemble(dl.derivative(dmr, z_fun, z_test), tensor=dmzr)
            dzq.axpy(self.sample_weights[k], dmzr)

        ystar_fun = vector2Function(self.ystar, self.pde.Vh[ADJOINT])
        dyr = dl.derivative(form, y_fun, ystar_fun)
        dyzr = self.model.generate_vector(CONTROL)
        dl.assemble(dl.derivative(dyr, z_fun, z_test), tensor=dyzr)
        dzq.axpy(1.0, dyzr)

        xstar_fun = vector2Function(self.xstar, self.pde.Vh[STATE])
        dxr = dl.derivative(form, x_fun, xstar_fun)
        dxzr = self.model.generate_vector(CONTROL)
        dl.assemble(dl.derivative(dxr, z_fun, z_test), tensor=dxzr)
        dzq.axpy(1.0, dxzr)

        for i in range(self.N_tr):
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
            ) = self.pde.gradientControl(
                self.x_all,
                self.xstar,
                self.ystar,
                self.xhat[i],
                self.xhatstar[i],
                self.mhat[i],
                self.mhatstar[i],
                self.yhat[i],
                self.yhatstar[i],
                self.qoi,
            )

            dzq.axpy(1.0, dyxzr)
            dzq.axpy(1.0, dymzr)
            dzq.axpy(1.0, dxyzr)
            dzq.axpy(1.0, dxxzr)
            dzq.axpy(1.0, dxmzr)
            dzq.axpy(1.0, dmmzr)
            dzq.axpy(1.0, dmyzr)
            dzq.axpy(1.0, dmxzr)
            # The terms below are all zero if QoI doesn not explicitly depends on m and z. 
            dzq.axpy(1.0, dmxzq)
            dzq.axpy(1.0, dmmzq)
            dzq.axpy(1.0, dxxzq)
            dzq.axpy(1.0, dxmzq)

        dz = AugmentedVector(self.model.generate_vector(CONTROL), copy_vector=False)
        dz.get_vector().axpy(1.0, dzq)
        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.get_vector().axpy(1.0, pen)

        dz.set_scalar(1.0 - self.sample_weight_sum)

        self.tgrad = time.time() - tgrad
        self.grad_cache = dz.copy()
        return dz, np.sqrt(dz.inner(dz))

    def costHessian(self, zt, zt_dir):
        """Fallback Hessian-vector product around the current linearization point.

        The user request focused on correcting the quadratic CVaR objective and
        its first derivative. For compatibility with the existing interface we
        retain the same local second-variation fallback used in the previous port.
        """
        self.hess_ncalls += 1
        self._copy_zt(zt)
        self.z_dir.zero()
        if isinstance(zt_dir, AugmentedVector):
            self.z_dir.axpy(1.0, zt_dir.get_vector())
        else:
            self.z_dir.axpy(1.0, zt_dir)

        if not self._objective_matches_current_zt():
            self.objective()
            self._cache_objective_zt()

        Vh = self.pde.Vh
        z_fun = vector2Function(self.z, Vh[CONTROL])
        z_dir_fun = vector2Function(self.z_dir, Vh[CONTROL])
        m_fun = vector2Function(self.m, Vh[PARAMETER])
        x_fun = vector2Function(self.x, Vh[STATE])
        y_fun = vector2Function(self.y, Vh[ADJOINT])

        r_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        x_trial = dl.TrialFunction(Vh[STATE])
        x_star = dl.Function(Vh[STATE])
        y_test = dl.TestFunction(Vh[ADJOINT])

        rx_form = dl.derivative(r_form, x_fun, x_trial)
        rxy_form = dl.derivative(rx_form, y_fun, y_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzy_form = dl.derivative(rz_form, y_fun, y_test)
        Ly_form = -rzy_form
        dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0)

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

        z_test = dl.TestFunction(Vh[CONTROL])
        ry_form = dl.derivative(r_form, y_fun, y_star)
        ryz_form = dl.derivative(ry_form, z_fun, z_test)

        rx_form = dl.derivative(r_form, x_fun, x_star)
        rxz_form = dl.derivative(rx_form, z_fun, z_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzz_form = dl.derivative(rz_form, z_fun, z_test)

        Hz = dl.assemble(ryz_form + rxz_form + rzz_form)

        if self.penalization is not None:
            dzzp = self.model.generate_vector(CONTROL)
            self.penalization.hessian(self.z, self.z_dir, dzzp)
            Hz.axpy(1.0, dzzp)

        if isinstance(zt_dir, AugmentedVector):
            Hz_aug = AugmentedVector(Hz, copy_vector=False)
            Hz_aug.set_scalar(0.0)
            return Hz_aug

        return Hz


class TaylorQuadraticCVaRControlCostFunctional(ControlCostFunctional):
    """Wrapper exposing the quadratic Taylor CVaR approximation."""

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
        return self._legacy.Q_0

    @property
    def cvar(self):
        return self._legacy.cvar

    @property
    def t_opt(self):
        return self._legacy.t_opt

    @property
    def d(self):
        return self._legacy.d

    @property
    def Q_surrogate(self):
        return self._legacy.Q_surrogate

    @property
    def Q_mc(self):
        return None

    def generate_vector(self, component="ALL"):
        if component == CONTROL:
            return AugmentedVector(self._legacy.model.generate_vector(CONTROL), copy_vector=False)
        return self._legacy.model.generate_vector(component)

    def cost(self, zt, order=0, FD_gradient_check=False):
        value = self._legacy.costValue(zt)
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

    def hessian(self, zhat, Hzhat):
        Hz = self._legacy.costHessian(self._legacy.zt, zhat)
        Hzhat.zero()
        Hzhat.axpy(1.0, Hz)


__all__ = ["TaylorQuadraticCVaRControlCostFunctional", "surrogate_cvar_from_samples"]
