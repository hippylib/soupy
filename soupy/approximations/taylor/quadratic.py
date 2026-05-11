"""Second-order (quadratic) Taylor approximation cost functional."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import dolfin as dl
from hippylib import Random, vector2Function, MultiVector
from hippylib.algorithms.randomizedEigensolver import doublePassG

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.reducedHessianSVD import ReducedHessianSVD
from ...modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL
from .settings import taylor_quadratic_settings


class _TaylorQuadraticLegacy:
    """Port of the legacy quadratic Taylor approximation with optional MC correction."""

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
        self.mhelp = model.generate_vector(PARAMETER)
        self.Hmhat1 = model.generate_vector(PARAMETER)
        self.Cdmq = model.generate_vector(PARAMETER)
        self.dmq = model.generate_vector(PARAMETER)  # Store for MC correction

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.N_tr = settings["N_tr"]
        self.beta = settings["beta"]
        self.seed = settings["seed"]
        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        try:
            self.correction = settings["correction"]
        except (KeyError, ValueError):
            self.correction = False

        try:
            self.N_mc = settings["N_mc"]
        except (KeyError, ValueError):
            self.N_mc = 0

        n_extra = self.N_mc if (self.correction and self.N_mc > 0) else 0
        self.n_modes = self.N_tr + n_extra
        self.xhat = [model.generate_vector(STATE) for _ in range(self.n_modes)]
        self.yhat = [model.generate_vector(STATE) for _ in range(self.n_modes)]
        self.xhatstar = [model.generate_vector(STATE) for _ in range(self.n_modes)]
        self.yhatstar = [model.generate_vector(STATE) for _ in range(self.n_modes)]
        self.mhat = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]
        self.mhatstar = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)
        self._omega_template = MultiVector(self.pde.generate_parameter(), self.N_tr)
        rand = Random(seed=self.seed)
        for i in range(self.N_tr):
            rand.normal(1.0, self._omega_template[i])

        # MPI setup
        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        # MC correction setup
        if self.correction and self.N_mc > 0:
            self.m_mc = []
            self.x_mc = []
            self.y_mc = []
            self.Q_mc = np.zeros(self.N_mc)
            randomGen = Random(myid=0, nproc=self.mpi_size)
            for _ in range(self.N_mc):
                noise = dl.Vector()
                prior.init_vector(noise, "noise")
                randomGen.normal(1.0, noise)

                sample = dl.Vector()
                prior.init_vector(sample, 1)
                prior.sample(noise, sample, add_mean=False)
                self.m_mc.append(sample)
                self.x_mc.append(self.pde.generate_state())
                self.y_mc.append(self.pde.generate_state())

            self.dmr = np.zeros(self.N_mc)
            self.dmmr = np.zeros(self.N_mc)
            self.const = np.zeros(self.N_mc)
            self.lin_diff_mean = np.zeros(self.N_mc)
            self.lin_diff_var = np.zeros(self.N_mc)
            self.quad_diff_mean = np.zeros(self.N_mc)
            self.quad_diff_var = np.zeros(self.N_mc)
            self.quad_fval_mean = np.zeros(self.N_mc)
        else:
            self.dmr = np.zeros(0)
            self.dmmr = np.zeros(0)
            self.const = np.zeros(0)
            self.lin_diff_mean = np.zeros(0)
            self.lin_diff_var = np.zeros(0)
            self.quad_diff_mean = np.zeros(0)
            self.quad_diff_var = np.zeros(0)
            self.quad_fval_mean = np.zeros(0)

        self.lin_mean = 0.0
        self.lin_var = 0.0

        self.quad_mean = 0.0
        self.quad_var = 0.0

        self.mean_diff = 0.0
        self.var_diff = 0.0
        self.Q_0 = 0.0

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

    def objectiveLinear(self):
        self.x_all[CONTROL] = self.z
        self.pde.solveFwd(self.x, self.x_all) # Inside this action, self.x is updated
        Q0 = self.qoi.cost(self.x_all)
        self.Q_0 = Q0
        self.x_all[STATE] = self.x

        rhs = self.model.generate_vector(STATE)
        self.qoi.adj_rhs(self.x_all, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs)
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False) # False means we do not use Gauss-newton approximation
        self.qoi.setLinearizationPoint(self.x_all)

        self.lin_mean = Q0
        self.x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        self.y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        self.m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        self.z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])

        form = self.pde.varf_handler(self.x_fun, self.m_fun, self.y_fun, self.z_fun)
        m_test = dl.TestFunction(self.pde.Vh[PARAMETER])
        self.dmq.zero()
        self.dmq.axpy(1.0, dl.assemble(dl.derivative(form, self.m_fun, m_test)))
        self.prior.Rsolver.solve(self.Cdmq, self.dmq)
        self.lin_var = Q0 ** 2 + self.dmq.inner(self.Cdmq)

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, self.m)
                m_i.axpy(1.0, self.m_mc[i])

                x_all_mc = [self.x_mc[i], m_i, self.y, self.z]
                self.pde.solveFwd(self.x_mc[i], x_all_mc)
                Q_i = self.qoi.cost([self.x_mc[i], m_i, self.y, self.z])
                self.Q_mc[i] = Q_i

                m_hat_fun = vector2Function(self.m_mc[i], self.pde.Vh[PARAMETER])
                dmr_i = dl.assemble(dl.derivative(form, self.m_fun, m_hat_fun))
                self.dmr[i] = dmr_i

                self.lin_diff_mean[i] = Q_i - (Q0 + dmr_i)
                self.lin_diff_var[i] = Q_i ** 2 - (Q0 + dmr_i) ** 2

        return Q0

    def objective(self):
        Q0 = self.objectiveLinear()

        omega = MultiVector(self.pde.generate_parameter(), self.N_tr)
        for i in range(self.N_tr):
            omega[i].zero()
            omega[i].axpy(1.0, self._omega_template[i])

        self.d, self.U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, self.N_tr, s=2)
        mean_quad_diff = self._mean_quad_diff()

        for i in range(self.N_tr):
            self.mhat[i].zero()
            self.mhat[i].axpy(1.0, self.U[i])

            _, xhat_i, yhat_i = self.HessianInner(self.mhat[i], self.mhat[i])
            self.xhat[i].zero()
            self.xhat[i].axpy(1.0, xhat_i)
            self.yhat[i].zero()
            self.yhat[i].axpy(1.0, yhat_i)

        self.quad_mean = Q0 + 0.5 * np.sum(self.d)
        self.quad_var = self.lin_var + 0.25 * (np.sum(self.d) ** 2) + 0.5 * np.sum(self.d ** 2) + Q0 * np.sum(self.d)

        # Currently, MC correction is never used. 
        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                dmmr_i, xhat_i, yhat_i = self.HessianInner(self.m_mc[i], self.m_mc[i])
                self.dmmr[i] = dmmr_i
                self.xhat[self.N_tr + i].zero()
                self.xhat[self.N_tr + i].axpy(1.0, xhat_i)
                self.yhat[self.N_tr + i].zero()
                self.yhat[self.N_tr + i].axpy(1.0, yhat_i)

                self.quad_diff_mean[i] = self.Q_mc[i] - (Q0 + self.dmr[i] + 0.5 * self.dmmr[i])
                self.quad_diff_var[i] = self.Q_mc[i] ** 2 - (Q0 + self.dmr[i] + 0.5 * self.dmmr[i]) ** 2
                self.quad_fval_mean[i] = self.Q_mc[i]

            mean_quad_diff = self._mean_quad_diff()
            for i in range(self.N_mc):
                self.const[i] = (
                    -0.5 / self.N_mc
                    - self.beta / self.N_mc * (Q0 + self.dmr[i] + 0.5 * self.dmmr[i])
                    + self.beta / self.N_mc * (self.quad_mean + mean_quad_diff)
                )

        for i in range(self.N_tr):
            self.mhatstar[i].zero()
            coeff = 0.5 + self.beta * self.d[i]
            if self.correction and self.N_mc > 0:
                coeff -= self.beta * mean_quad_diff
            self.mhatstar[i].axpy(coeff, self.U[i])

        self.mean_diff = mean_quad_diff # This is zero is there's no correction
        self.var_diff = self._mean_quad_var_diff() # This is zero if there's no correction
        corrected_mean = self.quad_mean + self.mean_diff
        corrected_var = self.quad_var + self.var_diff
        return corrected_mean + self.beta * (corrected_var - corrected_mean ** 2)

    def _mean_quad_diff(self):
        if self.correction and self.N_mc > 0:
            return float(np.mean(self.quad_diff_mean))
        return 0.0

    def _mean_quad_var_diff(self):
        if self.correction and self.N_mc > 0:
            return float(np.mean(self.quad_diff_var))
        return 0.0

    def HessianInner(self, mhat1, mhat2):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()

        self.pde.apply_ij(ADJOINT, PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False)

        self.rhs_adj.zero()
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1.0, self.rhs_adj2)
        self.qoi.apply_ij(STATE, STATE, xhat, self.rhs_adj3)
        self.rhs_adj.axpy(1.0, self.rhs_adj3)
        self.qoi.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj4)
        self.rhs_adj.axpy(1.0, self.rhs_adj4)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True)

        self.pde.apply_ij(PARAMETER, PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.pde.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.qoi.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)
        self.qoi.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1.0, self.mhelp)

        return mhat2.inner(self.Hmhat1), xhat, yhat

    def costValue(self, z):
        self.func_ncalls += 1
        self._copy_z(z)
        objective = self.objective()
        self._cache_objective_z()
        self.grad_cache = None
        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalty

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

        dmr = dl.derivative(form, m_fun, mhat_fun)
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

        return dmxr, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr, dxmxq, dmmxq, dmxxq

    def solveAdjIncrementalAdj(self):
        mean_quad_diff = self._mean_quad_diff()
        for i in range(self.N_tr):
            self.xhatstar[i].zero()
            coeff = 0.5 + self.beta * self.d[i]
            if self.correction and self.N_mc > 0:
                coeff -= self.beta * mean_quad_diff
            self.xhatstar[i].axpy(coeff, self.xhat[i])

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                idx = self.N_tr + i
                dmyr = self.pde.forSolveAdjIncrementalAdj(self.x_all, self.m_mc[i])
                xhatstarrhs = self.pde.generate_state()
                xhatstarrhs.axpy(self.const[i], dmyr)
                self.pde.solveIncremental(self.xhatstar[idx], -xhatstarrhs, False)

    def solveAdjIncrementalFwd(self):
        mean_quad_diff = self._mean_quad_diff()
        for i in range(self.N_tr):
            self.yhatstar[i].zero()
            coeff = 0.5 + self.beta * self.d[i]
            if self.correction and self.N_mc > 0:
                coeff -= self.beta * mean_quad_diff
            self.yhatstar[i].axpy(coeff, self.yhat[i])

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                idx = self.N_tr + i
                dmxr, dxxr, dxxq = self.pde.forSolveAdjIncrementalFwd(
                    self.x_all, self.m_mc[i], self.xhatstar[idx], self.qoi
                )
                yhatstarrhs = self.pde.generate_state()
                yhatstarrhs.axpy(self.const[i], dmxr)
                yhatstarrhs.axpy(1.0, dxxr)
                yhatstarrhs.axpy(1.0, dxxq)
                self.pde.solveIncremental(self.yhatstar[idx], -yhatstarrhs, True)

    def solveAdjAdj(self):
        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        y_test = dl.TestFunction(self.pde.Vh[ADJOINT])

        xstarrhs = self.pde.generate_state()
        Cdmq_fun = vector2Function(self.Cdmq, self.pde.Vh[PARAMETER])
        dmr = dl.derivative(form, m_fun, Cdmq_fun)
        dmyr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmr, y_fun, y_test), tensor=dmyr)
        [bc.apply(dmyr) for bc in self.pde.bc0]
        xstarrhs.axpy(2.0 * self.beta, dmyr)

        for i in range(self.N_tr):
            dmyr, dxxyr, dxmyr, dmmyr, dmxyr = self._forSolveAdjAdj(
                self.xhat[i], self.xhatstar[i], self.mhat[i], self.mhatstar[i]
            )
            xstarrhs.axpy(1.0, dxxyr)
            xstarrhs.axpy(1.0, dxmyr)
            xstarrhs.axpy(1.0, dmmyr)
            xstarrhs.axpy(1.0, dmxyr)

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                idx = self.N_tr + i
                dmyr, dxxyr, dxmyr, dmmyr, dmxyr = self._forSolveAdjAdj(
                    self.xhat[idx], self.xhatstar[idx], self.m_mc[i], self.m_mc[i]
                )
                xstarrhs.axpy(1.0, dxxyr)
                xstarrhs.axpy(1.0, dxmyr)
                xstarrhs.axpy(2.0 * self.const[i], dmyr)
                xstarrhs.axpy(self.const[i], dmmyr)
                xstarrhs.axpy(self.const[i], dmxyr)

        self.pde.solveIncremental(self.xstar, -xstarrhs, False)

    def solveAdjFwd(self):
        ystarrhs = self.pde.generate_state()
        ystarrhsqoi = self.pde.generate_state()
        self.qoi.grad(STATE, self.x_all, ystarrhsqoi)
        [bc.apply(ystarrhsqoi) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, ystarrhsqoi)

        ystarrhspde = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1.0, ystarrhspde)
        self.pde.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1.0, ystarrhspde)
        [bc.apply(ystarrhs) for bc in self.pde.bc0]

        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        x_test = dl.TestFunction(self.pde.Vh[STATE])

        Cdmq_fun = vector2Function(self.Cdmq, self.pde.Vh[PARAMETER])
        dmr = dl.derivative(form, m_fun, Cdmq_fun)
        dmxr = self.pde.generate_state()
        dl.assemble(dl.derivative(dmr, x_fun, x_test), tensor=dmxr)
        [bc.apply(dmxr) for bc in self.pde.bc0]
        ystarrhs.axpy(2.0 * self.beta, dmxr)

        for i in range(self.N_tr):
            dmxr, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr, _, _, _ = self._forSolveAdjFwd(
                self.xhat[i], self.xhatstar[i], self.mhat[i], self.mhatstar[i], self.yhat[i], self.yhatstar[i]
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

        if self.correction and self.N_mc > 0:
            ystarrhs.axpy(-2.0 * self.beta * self._mean_quad_diff(), ystarrhsqoi)
            for i in range(self.N_mc):
                idx = self.N_tr + i
                dmxr, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr, _, _, _ = self._forSolveAdjFwd(
                    self.xhat[idx], self.xhatstar[idx], self.m_mc[i], self.m_mc[i], self.yhat[idx], self.yhatstar[idx]
                )
                ystarrhs.axpy(1.0, dyxxr)
                ystarrhs.axpy(1.0, dymxr)
                ystarrhs.axpy(1.0, dxyxr)
                ystarrhs.axpy(1.0, dxxxr)
                ystarrhs.axpy(1.0, dxmxr)
                ystarrhs.axpy(1.0, dxxxq)
                ystarrhs.axpy(2.0 * self.const[i], dmxr)
                ystarrhs.axpy(2.0 * self.const[i], ystarrhsqoi)
                ystarrhs.axpy(self.const[i], dmmxr)
                ystarrhs.axpy(self.const[i], dmxxr)
                ystarrhs.axpy(self.const[i], dmyxr)

        self.pde.solveIncremental(self.ystar, -ystarrhs, True)

    def solve_ymc(self):
        if not (self.correction and self.N_mc > 0):
            return

        mean_quad_diff = self._mean_quad_diff()
        for i in range(self.N_mc):
            m_i = self.pde.generate_parameter()
            m_i.axpy(1.0, self.m)
            m_i.axpy(1.0, self.m_mc[i])

            x_all_i = [self.x_mc[i], m_i, self.y, self.z]
            rhs = self.pde.generate_state()
            self.qoi.adj_rhs([self.x_mc[i], m_i, self.y, self.z], rhs)
            [bc.apply(rhs) for bc in self.pde.bc0]
            rhs[:] *= (
                1.0
                / self.N_mc
                * (1.0 + 2.0 * self.beta * self.quad_fval_mean[i] - 2.0 * self.beta * (self.quad_mean + mean_quad_diff))
            )

            self.pde.solveAdj(self.y_mc[i], x_all_i, rhs)

    def costGradient(self, z):
        self.grad_ncalls += 1
        self._copy_z(z)
        if not self._objective_matches_current_z():
            self.objective()
            self._cache_objective_z()

        self.solveAdjIncrementalAdj()
        self.solveAdjIncrementalFwd()
        self.solveAdjAdj()
        self.solveAdjFwd()
        self.solve_ymc()

        dzq = self.model.generate_vector(CONTROL)

        x_fun = vector2Function(self.x, self.pde.Vh[STATE])
        y_fun = vector2Function(self.y, self.pde.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.pde.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.pde.Vh[CONTROL])
        form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        z_test = dl.TestFunction(self.pde.Vh[CONTROL])

        Cdmq_fun = vector2Function(self.Cdmq, self.pde.Vh[PARAMETER])
        dmr = dl.derivative(form, m_fun, Cdmq_fun)
        dmzr = self.model.generate_vector(CONTROL)
        dl.assemble(dl.derivative(dmr, z_fun, z_test), tensor=dmzr)
        dzq.axpy(2.0 * self.beta, dmzr)

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

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, self.m)
                m_i.axpy(1.0, self.m_mc[i])
                m_fun_i = vector2Function(m_i, self.pde.Vh[PARAMETER])
                x_fun_i = vector2Function(self.x_mc[i], self.pde.Vh[STATE])
                y_fun_i = vector2Function(self.y_mc[i], self.pde.Vh[ADJOINT])
                form_i = self.pde.varf_handler(x_fun_i, m_fun_i, y_fun_i, z_fun)

                dyr_i = dl.derivative(form_i, y_fun_i, y_fun_i)
                dyzr_i = self.model.generate_vector(CONTROL)
                dl.assemble(dl.derivative(dyr_i, z_fun, z_test), tensor=dyzr_i)
                dzq.axpy(1.0, dyzr_i)

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
            dzq.axpy(1.0, dmxzr)
            dzq.axpy(1.0, dmyzr)
            dzq.axpy(1.0, dmxzq)
            dzq.axpy(1.0, dmmzq)
            dzq.axpy(1.0, dxxzq)
            dzq.axpy(1.0, dxmzq)

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                idx = self.N_tr + i
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
                    self.xhat[idx],
                    self.xhatstar[idx],
                    self.m_mc[i],
                    self.m_mc[i],
                    self.yhat[idx],
                    self.yhatstar[idx],
                    self.qoi,
                )

                mhat_fun = vector2Function(self.m_mc[i], self.pde.Vh[PARAMETER])
                dmr_i = dl.derivative(form, m_fun, mhat_fun)
                dmzr_i = self.model.generate_vector(CONTROL)
                dl.assemble(dl.derivative(dmr_i, z_fun, z_test), tensor=dmzr_i)

                dzq.axpy(1.0, dyxzr)
                dzq.axpy(1.0, dymzr)
                dzq.axpy(1.0, dxyzr)
                dzq.axpy(1.0, dxxzr)
                dzq.axpy(1.0, dxmzr)
                dzq.axpy(2.0 * self.const[i], dmzr_i)
                dzq.axpy(self.const[i], dmmzr)
                dzq.axpy(self.const[i], dmxzr)
                dzq.axpy(self.const[i], dmyzr)
                dzq.axpy(self.const[i], dmxzq)
                dzq.axpy(self.const[i], dmmzq)
                dzq.axpy(self.const[i], dxxzq)
                dzq.axpy(self.const[i], dxmzq)

        dz = self.model.generate_vector(CONTROL)
        dz.axpy(1.0, dzq)
        if self.penalization is not None:
            pen = self.model.generate_vector(CONTROL)
            self.penalization.grad(self.z, pen)
            dz.axpy(1.0, pen)

        norm = np.sqrt(dz.inner(dz))
        self.grad_cache = dz.copy()
        return dz, norm

    def costHessian(self, z, z_dir):
        """Compute Hessian-vector product Hz = H * z_dir.

        The Hessian of the quadratic Taylor approximation w.r.t. control
        follows the same structure as the linear case since the gradient
        depends on the control through the same mechanism.
        """
        self.z.zero()
        self.z.axpy(1.0, z)
        self.z_dir.zero()
        self.z_dir.axpy(1.0, z_dir)

        # Ensure objective has been computed to set linearization point
        self.objective()

        Vh = self.pde.Vh

        z_fun = vector2Function(self.z, Vh[CONTROL])
        z_dir_fun = vector2Function(self.z_dir, Vh[CONTROL])
        m_fun = vector2Function(self.m, Vh[PARAMETER])
        x_fun = vector2Function(self.x, Vh[STATE])
        y_fun = vector2Function(self.y, Vh[ADJOINT])

        r_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        # Solve incremental forward: A * x_star = -dA/dz * z_dir * y
        x_trial = dl.TrialFunction(Vh[STATE])
        x_star = dl.Function(Vh[STATE])
        y_test = dl.TestFunction(Vh[ADJOINT])

        rx_form = dl.derivative(r_form, x_fun, x_trial)
        rxy_form = dl.derivative(rx_form, y_fun, y_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzy_form = dl.derivative(rz_form, y_fun, y_test)
        Ly_form = -rzy_form

        dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0)

        # Solve incremental adjoint: A^T * y_star = -(d^2L/dx^2 * x_star + d^2A/dxdz * z_dir)
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

        # Assemble Hessian action: Hz = d^2L/dydz * y_star + d^2L/dxdz * x_star + d^2L/dz^2 * z_dir
        z_test = dl.TestFunction(Vh[CONTROL])
        ry_form = dl.derivative(r_form, y_fun, y_star)
        ryz_form = dl.derivative(ry_form, z_fun, z_test)

        rx_form = dl.derivative(r_form, x_fun, x_star)
        rxz_form = dl.derivative(rx_form, z_fun, z_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzz_form = dl.derivative(rz_form, z_fun, z_test)

        Lz_form = ryz_form + rxz_form + rzz_form

        Hz = dl.assemble(Lz_form)

        # Add penalization Hessian contribution
        if self.penalization is not None:
            dzzp = self.model.generate_vector(CONTROL)
            self.penalization.hessian(self.z, self.z_dir, dzzp)
            Hz.axpy(1.0, dzzp)

        return Hz


class TaylorQuadraticControlCostFunctional(ControlCostFunctional):
    """Wrapper exposing the second-order Taylor approximation."""

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.settings = taylor_quadratic_settings(settings)
        self._legacy = _TaylorQuadraticLegacy(self.settings, control_model, prior, penalization, tol)

    def generate_vector(self, component="ALL"):
        return self._legacy.model.generate_vector(component)

    def cost(self, z, order=0, FD_gradient_check=False):
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
