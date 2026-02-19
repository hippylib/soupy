"""Second-order (quadratic) Taylor approximation cost functional."""

from __future__ import annotations

import time
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

        self.xstar = model.generate_vector(STATE)
        self.ystar = model.generate_vector(STATE)
        self.mhelp = model.generate_vector(PARAMETER)
        self.Cdmq = model.generate_vector(PARAMETER)
        self.dmq = model.generate_vector(PARAMETER)  # Store for MC correction

        self.func_ncalls = 0
        self.grad_ncalls = 0

        self.N_tr = settings["N_tr"]
        self.beta = settings["beta"]
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

        self.xhat = [model.generate_vector(STATE) for _ in range(self.N_tr)]
        self.yhat = [model.generate_vector(PARAMETER) for _ in range(self.N_tr)]

        self.H = ReducedHessianSVD(self.pde, self.qoi, tol)

        # MPI setup
        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        # MC correction setup
        if self.correction and self.N_mc > 0:
            self.m_mc = []
            self.x_mc = []
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

            self.quad_diff_mean = np.zeros(self.N_mc)
            self.quad_diff_var = np.zeros(self.N_mc)

        self.lin_mean = 0.0
        self.lin_var = 0.0

        self.quad_mean = 0.0
        self.quad_var = 0.0

        self.mean_diff = 0.0
        self.var_diff = 0.0

    def objectiveLinear(self):
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

        return Q0

    def objective(self):
        Q0 = self.objectiveLinear()

        omega = MultiVector(self.pde.generate_parameter(), self.N_tr + 5)
        rand = Random()
        for i in range(self.N_tr + 5):
            rand.normal(1.0, omega[i])

        self.d, self.U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, omega, self.N_tr, s=1)

        for i in range(self.N_tr):
            self.xhat[i].zero()
            self.yhat[i].zero()
            self.pde.apply_ij(ADJOINT, PARAMETER, self.U[i], self.rhs_fwd)
            self.pde.solveIncremental(self.xhat[i], -self.rhs_fwd, False)
            self.pde.apply_ij(PARAMETER, STATE, self.xhat[i], self.yhat[i])

        self.quad_mean = Q0 + 0.5 * np.sum(self.d)
        self.quad_var = self.lin_var + 0.25 * (np.sum(self.d) ** 2) + 0.5 * np.sum(self.d ** 2) + Q0 * np.sum(self.d)

        # Monte Carlo correction
        self.mean_diff = 0.0
        self.var_diff = 0.0

        if self.correction and self.N_mc > 0:
            # Pre-compute R * m_sample for each MC sample
            # The eigenvectors U are R-orthonormal (U^T R U = I), so we need
            # (R U_k) · m to get standard normal projections when m ~ N(0, R^{-1})
            R_m_samples = []
            for i in range(self.N_mc):
                R_m = self.pde.generate_parameter()
                self.prior.R.mult(self.m_mc[i], R_m)
                R_m_samples.append(R_m)

            for i in range(self.N_mc):
                m_sample = self.m_mc[i]  # Zero-mean sample (m_i - m_bar)

                # Compute m_i = m_bar + m_sample
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, self.m)
                m_i.axpy(1.0, m_sample)

                # Solve forward problem at m_i
                x_all_mc = [self.x_mc[i], m_i, self.y, self.z]
                self.pde.solveFwd(self.x_mc[i], x_all_mc)

                # Compute true QoI value Q(m_i)
                Q_i = self.qoi.cost([self.x_mc[i], m_i, self.y, self.z])
                self.Q_mc[i] = Q_i

                # Compute linear Taylor term: dQ/dm · (m_i - m_bar)
                linear_term = self.dmq.inner(m_sample)

                # Compute quadratic Taylor term: 0.5 * sum_k d_k * (U_k · R * m_sample)^2
                # Using R-weighted inner product to match analytical variance formula
                quad_term = 0.0
                for k in range(self.N_tr):
                    Uk_dot_Rm = self.U[k].inner(R_m_samples[i])
                    quad_term += self.d[k] * Uk_dot_Rm ** 2
                quad_term *= 0.5

                # Taylor approximation: Q0 + linear + quadratic
                Q_taylor = Q0 + linear_term + quad_term

                # Store corrections
                self.quad_diff_mean[i] = Q_i - Q_taylor
                self.quad_diff_var[i] = Q_i ** 2 - Q_taylor ** 2

            self.mean_diff = np.mean(self.quad_diff_mean)
            self.var_diff = np.mean(self.quad_diff_var)

        # Corrected mean and variance
        corrected_mean = self.quad_mean + self.mean_diff
        corrected_var = self.quad_var + self.var_diff

        return corrected_mean + self.beta * (corrected_var - corrected_mean ** 2)

    def costValue(self, z):
        self.func_ncalls += 1
        self.z.zero()
        self.z.axpy(1.0, z)
        objective = self.objective()
        penalty = 0.0 if self.penalization is None else self.penalization.cost(self.z)
        return objective + penalty

    def costGradient(self, z):
        self.grad_ncalls += 1
        self.z.zero()
        self.z.axpy(1.0, z)
        self.objective()

        dz = self.model.generate_vector(CONTROL)
        self.pde.evalGradientControl(self.x_all, dz)
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
