"""First-order (linear) Taylor approximation cost functional."""

from __future__ import annotations

import time
from typing import Optional, Union

import dolfin as dl
import numpy as np
from hippylib import Random, vector2Function

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .settings import taylor_linear_settings


OPTIMIZATION = CONTROL  # Legacy alias used in the ported implementation


def _dlversion():
    return (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)


class _TaylorLinearLegacy:
    """Direct port of the legacy linear Taylor cost functional implementation."""

    def __init__(self, parameter, Vh, pde, qoi, prior, penalization, tol=1e-9):

        self.parameter = parameter
        self.Vh = Vh
        self.pde = pde
        self.qoi = qoi
        self.prior = prior
        self.penalization = penalization
        self.z = pde.generate_control()
        self.m = prior.mean
        self.tol = tol

        self.x = pde.generate_state()
        self.y = pde.generate_state()
        self.x_all = [self.x, self.m, self.y, self.z]

        self.z_dir = pde.generate_control()
        self.z_diff = pde.generate_control()
        self.dz = None

        self.xstar = pde.generate_state()
        self.ystar = pde.generate_state()

        self.Cdmq = pde.generate_parameter()

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0
        self.ncalls = 0

        self.correction = parameter["correction"]
        self.N_mc = parameter["N_mc"]
        self.beta = parameter["beta"]
        try:
            self.verbose = parameter["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        self.lin_mean = 0.0
        self.lin_var = 0.0
        self.lin_diff_mean = np.zeros(self.N_mc)
        self.lin_fval_mean = np.zeros(self.N_mc)
        self.lin_diff_var = np.zeros(self.N_mc)
        self.lin_fval_var = np.zeros(self.N_mc)

        self.dmr = np.zeros(self.N_mc)

        self.sign = 1.0

        self.quad_mean = 0.0
        self.quad_var = 0.0
        self.quad_diff_mean = np.zeros(self.N_mc)
        self.quad_fval_mean = np.zeros(self.N_mc)
        self.quad_diff_var = np.zeros(self.N_mc)
        self.quad_fval_var = np.zeros(self.N_mc)

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()

        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())
        if self.mpi_size > 1:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD

        if self.correction:
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

                self.x_mc.append(pde.generate_state())
                self.y_mc.append(pde.generate_state())

        self.tobj = 0.0
        self.tgrad = 0.0
        self.trand = 0.0

        self.Msolver = None

    def objective(self):
        self.x_all[OPTIMIZATION] = self.z
        self.pde.solveFwd(self.x, self.x_all)
        Q_0 = self.qoi.cost(self.x_all)
        self.x_all[STATE] = self.x

        rhs = self.pde.generate_state()
        self.qoi.adj_rhs(self.x_all, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs)
        self.x_all[ADJOINT] = self.y

        self.pde.setLinearizationPoint(self.x_all, False)
        self.qoi.setLinearizationPoint(self.x_all)

        self.lin_mean = Q_0
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        m_test = dl.TestFunction(self.Vh[PARAMETER])
        dmq = dl.assemble(dl.derivative(f_form, m_fun, m_test))
        Cdmq = self.pde.generate_parameter()
        self.prior.Rsolver.solve(Cdmq, dmq)
        self.Cdmq = Cdmq
        self.lin_var = Q_0 ** 2 + dmq.inner(Cdmq)

        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m = self.m_mc[i]
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, m)
                m_i.axpy(1.0, self.m)
                x_all = [self.x, m_i, self.y, self.z]
                x = self.pde.generate_state()
                self.pde.solveFwd(x, x_all)
                self.x_mc[i] = x
                Q_i = self.qoi.cost([x, m_i, self.y_mc[i], self.z])
                self.Q_mc[i] = Q_i

                f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.assemble(dl.derivative(f_form, m_fun, m_hat))
                self.dmr[i] = dmr

                self.lin_diff_mean[i] = Q_i - (Q_0 + dmr)
                self.lin_fval_mean[i] = Q_i
                self.lin_diff_var[i] = Q_i ** 2 - (Q_0 + dmr) ** 2
                self.lin_fval_var[i] = Q_i ** 2

        mean_diff = np.mean(self.lin_diff_mean) if self.N_mc > 0 else 0.0
        var_diff = np.mean(self.lin_diff_var) if self.N_mc > 0 else 0.0

        cost = (
            self.lin_mean
            + mean_diff
            + self.beta * (self.lin_var + var_diff)
            - self.beta * (self.lin_mean + mean_diff) ** 2
        )

        if self.verbose and self.mpi_rank == 0:
            print("  [Linear Taylor] mean={:.3e}, var={:.3e}".format(
                self.lin_mean, self.lin_var - self.lin_mean ** 2))
            if self.correction and self.N_mc > 0:
                print("                  mean_diff={:.3e}, var_diff={:.3e}".format(
                    mean_diff, var_diff))

        return cost

    def costValue(self, z):
        self.func_ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0] : idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)

        tobj = time.time()

        objective = self.objective()

        self.tobj = time.time() - tobj

        penalization = 0.0
        if self.penalization is not None:
            penalization = self.penalization.cost(self.z)

        cost = objective + penalization

        if self.verbose and self.mpi_rank == 0:
            print("  [Cost eval #{}] cost={:.6e}, obj={:.3e}, pen={:.3e}".format(
                self.func_ncalls, cost, objective, penalization))

        return cost

    def solve_xstar(self):
        """Solve the incremental forward problem for the variance gradient.

        The RHS comes from differentiating the variance term
        Var_lin = <dmq, C*dmq> w.r.t. the state through the adjoint.
        """
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        y_test = dl.TestFunction(self.Vh[ADJOINT])
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        xstarrhs = self.pde.generate_state()

        # Variance term: 2*beta * d^2r/(dm dy) * C*dmq
        dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmyrCdmq = dl.assemble(dl.derivative(dmrCdmq, y_fun, y_test))
        [bc.apply(dmyrCdmq) for bc in self.pde.bc0]
        xstarrhs.axpy(2 * self.beta, dmyrCdmq)

        # MC correction terms
        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.derivative(f_form, m_fun, m_hat)
                dmyr = dl.derivative(dmr, y_fun, y_test)
                dmyr_ass = dl.assemble(dmyr)
                [bc.apply(dmyr_ass) for bc in self.pde.bc0]
                xstarrhs.axpy(
                    -1.0 / self.N_mc
                    - 2.0 * self.beta / self.N_mc * (self.lin_mean + self.dmr[i])
                    + 2.0 * self.beta / self.N_mc * (self.lin_mean + np.mean(self.lin_diff_mean)),
                    dmyr_ass,
                )

        xstar = self.pde.generate_state()
        self.pde.solveIncremental(xstar, -xstarrhs, False)
        self.xstar = xstar

    def solve_ystar(self):
        """Solve the incremental adjoint problem for the variance gradient.

        The RHS includes contributions from the mean gradient, the variance
        derivative w.r.t. the state, and the incremental forward solution.
        """
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        x_test = dl.TestFunction(self.Vh[STATE])
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        ystarrhs = self.pde.generate_state()

        # Mean gradient: dQ/dx
        dxq = self.pde.generate_state()
        self.qoi.grad(STATE, self.x_all, dxq)
        [bc.apply(dxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxq)

        # Variance term: 2*beta * d^2r/(dm dx) * C*dmq
        dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmxrCdmq = dl.assemble(dl.derivative(dmrCdmq, x_fun, x_test))
        [bc.apply(dmxrCdmq) for bc in self.pde.bc0]
        ystarrhs.axpy(2 * self.beta, dmxrCdmq)

        # Incremental forward contribution: d^2r/(dx dx) * xstar
        xstar_fun = vector2Function(self.xstar, self.Vh[STATE])
        dxr = dl.derivative(f_form, x_fun, xstar_fun)
        dxxr = dl.assemble(dl.derivative(dxr, x_fun, x_test))
        [bc.apply(dxxr) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxr)

        # QoI Hessian contribution: d^2Q/(dx dx) * xstar
        dxxq = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, self.xstar, dxxq)
        [bc.apply(dxxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxq)

        # MC correction terms
        if self.correction and self.N_mc > 0:
            ystarrhs.axpy(2.0 * self.beta * self.lin_mean, dxq)
            for i in range(self.N_mc):
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.derivative(f_form, m_fun, m_hat)
                dmxr = dl.derivative(dmr, x_fun, x_test)
                dmxr_ass = dl.assemble(dmxr)
                [bc.apply(dmxr_ass) for bc in self.pde.bc0]
                ystarrhs.axpy(
                    -1.0 / self.N_mc
                    - 2.0 * self.beta / self.N_mc * (self.lin_mean + self.dmr[i]),
                    dxq,
                )
                ystarrhs.axpy(
                    -1.0 / self.N_mc
                    - 2.0 * self.beta / self.N_mc * (self.lin_mean + self.dmr[i])
                    + 2.0 * self.beta / self.N_mc * (self.lin_mean + np.mean(self.lin_diff_mean)),
                    dmxr_ass,
                )

        ystar = self.pde.generate_state()
        self.pde.solveIncremental(ystar, -ystarrhs, True)
        self.ystar = ystar

    def solve_ymc(self):
        """Solve MC adjoint problems for correction terms."""
        for i in range(self.N_mc):
            m = self.m_mc[i]
            m_i = self.pde.generate_parameter()
            m_i.axpy(1.0, m)
            m_i.axpy(1.0, self.m)
            x_all = [self.x_mc[i], m_i, self.y, self.z]

            rhs = self.pde.generate_state()
            self.qoi.adj_rhs(x_all, rhs)
            [bc.apply(rhs) for bc in self.pde.bc0]

            rhs[:] *= (
                1.0 / self.N_mc
                * (
                    1.0
                    + 2.0 * self.beta * self.lin_fval_mean[i]
                    - 2.0 * self.beta * (self.lin_mean + np.mean(self.lin_diff_mean))
                )
            )

            y_mc = self.pde.generate_state()
            self.pde.solveAdj(y_mc, x_all, rhs)
            self.y_mc[i] = y_mc

    def costGradient(self, z):
        self.grad_ncalls += 1
        self.ncalls = 0

        if isinstance(z, np.ndarray):
            zprev = self.z.gather_on_zero()
            if self.mpi_size > 1:
                zprev = self.comm.bcast(zprev, root=0)

            if np.linalg.norm(z - zprev) > 1e-15:
                self.costValue(z)

        else:
            self.z_diff.zero()
            self.z_diff.axpy(1.0, self.z)
            self.z_diff.axpy(-1.0, z)
            if self.z_diff.inner(self.z_diff) > 1e-20:
                self.costValue(z)

        tgrad = time.time()

        # Solve incremental problems for variance gradient
        if self.correction and self.N_mc > 0:
            self.solve_ymc()
        self.solve_xstar()
        self.solve_ystar()

        # Assemble gradient
        dzq = self.pde.generate_control()
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        # Direct variance derivative: 2*beta * d^2r/(dm dz) * C*dmq
        dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmzrCdmq = dl.assemble(dl.derivative(dmrCdmq, z_fun, z_test))
        dzq.axpy(2 * self.beta, dmzrCdmq)

        # MC correction for direct variance derivative
        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.derivative(f_form, m_fun, m_hat)
                dmzr = dl.derivative(dmr, z_fun, z_test)
                dmzr_ass = dl.assemble(dmzr)

                dzq.axpy(
                    -1.0 / self.N_mc
                    - 2.0 * self.beta / self.N_mc * (self.lin_mean + self.dmr[i])
                    + 2.0 * self.beta / self.N_mc * (self.lin_mean + np.mean(self.lin_diff_mean)),
                    dmzr_ass,
                )

        # Contribution from incremental adjoint (ystar)
        ystar_fun = vector2Function(self.ystar, self.Vh[ADJOINT])
        dyr = dl.derivative(f_form, y_fun, ystar_fun)
        dyzr = dl.assemble(dl.derivative(dyr, z_fun, z_test))
        dzq.axpy(1.0, dyzr)

        # Contribution from incremental forward (xstar)
        xstar_fun = vector2Function(self.xstar, self.Vh[STATE])
        dxr = dl.derivative(f_form, x_fun, xstar_fun)
        dxzr = dl.assemble(dl.derivative(dxr, z_fun, z_test))
        dzq.axpy(1.0, dxzr)

        # MC adjoint contributions
        if self.correction and self.N_mc > 0:
            for i in range(self.N_mc):
                m = self.m_mc[i]
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.0, m)
                m_i.axpy(1.0, self.m)
                m_fun_i = vector2Function(m_i, self.Vh[PARAMETER])
                x_fun_i = vector2Function(self.x_mc[i], self.Vh[STATE])
                y_fun_i = vector2Function(self.y_mc[i], self.Vh[ADJOINT])
                f_form_i = self.pde.varf_handler(x_fun_i, m_fun_i, y_fun_i, z_fun)

                dyr_i = dl.derivative(f_form_i, y_fun_i, y_fun_i)
                dyzr_i = dl.assemble(dl.derivative(dyr_i, z_fun, z_test))
                dzq.axpy(1.0, dyzr_i)

        # Assemble total gradient = objective gradient + penalization gradient
        dz = self.pde.generate_control()
        dz.axpy(1.0, dzq)

        if self.penalization is not None:
            dzp = self.pde.generate_control()
            self.penalization.grad(self.z, dzp)
            dz.axpy(1.0, dzp)

        self.tgrad = time.time() - tgrad

        return dz, np.sqrt(dz.inner(dz))

    def costHessian(self, z, z_dir, FD=False):
        self.hess_ncalls += 1
        self.ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0] : idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)
        if isinstance(z_dir, np.ndarray):
            idx = self.z_dir.local_range()
            self.z_dir.set_local(z_dir[idx[0] : idx[1]])
        else:
            self.z_dir.zero()
            self.z_dir.axpy(1.0, z_dir)

        if FD:
            epsilon = 1e-6
            if isinstance(z, np.ndarray):
                z1 = z - epsilon * z_dir
                z2 = z + epsilon * z_dir
            else:
                z1 = z.copy()
                z1.axpy(-epsilon, z_dir)
                z2 = z.copy()
                z2.axpy(epsilon, z_dir)

            self.costValue(z1)
            dz1, _ = self.costGradient(z1)
            self.costValue(z2)
            dz2, _ = self.costGradient(z2)

            if isinstance(z, np.ndarray):
                Hz = (dz2 - dz1) / (2 * epsilon)
            else:
                dz2.axpy(-1.0, dz1)
                Hz = dz2
                Hz[:] = Hz[:] / (2 * epsilon)

            return Hz

        thess = time.time()

        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        z_dir_fun = vector2Function(self.z_dir, self.Vh[OPTIMIZATION])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])

        r_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)

        x_trial = dl.TrialFunction(self.Vh[STATE])
        x_star = dl.Function(self.Vh[STATE])
        y_test = dl.TestFunction(self.Vh[ADJOINT])

        rx_form = dl.derivative(r_form, x_fun, x_trial)
        rxy_form = dl.derivative(rx_form, y_fun, y_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzy_form = dl.derivative(rz_form, y_fun, y_test)
        Ly_form = -rzy_form

        dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0)

        x_test = dl.TestFunction(self.Vh[STATE])
        y_trial = dl.TrialFunction(self.Vh[ADJOINT])
        y_star = dl.Function(self.Vh[ADJOINT])

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

        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        ry_form = dl.derivative(r_form, y_fun, y_star)
        ryz_form = dl.derivative(ry_form, z_fun, z_test)

        rx_form = dl.derivative(r_form, x_fun, x_star)
        rxz_form = dl.derivative(rx_form, z_fun, z_test)

        rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
        rzz_form = dl.derivative(rz_form, z_fun, z_test)

        Lz_form = ryz_form + rxz_form + rzz_form

        Hz = dl.assemble(Lz_form)

        if self.penalization is not None:
            dzzp = self.pde.generate_control()
            self.penalization.hessian(self.z, self.z_dir, dzzp)
            Hz.axpy(1.0, dzzp)

        return Hz


class TaylorLinearControlCostFunctional(ControlCostFunctional):
    """Wrapper exposing the legacy linear Taylor approximation via modern APIs."""

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, "ParameterList"]] = None,
        tol=1e-9,
    ):
        self.model = control_model
        self.prior = prior
        self.penalization = penalization
        self.settings = taylor_linear_settings(settings)

        self._legacy = _TaylorLinearLegacy(
            self.settings,
            self.model.problem.Vh,
            self.model.problem,
            self.model.qoi,
            prior,
            penalization,
            tol=tol,
        )

        self._grad_cache = None

    def generate_vector(self, component="ALL"):
        return self.model.generate_vector(component)

    def cost(self, z, order=0, FD_gradient_check=False):
        value = self._legacy.costValue(z)
        self._grad_cache = None
        if order >= 1:
            grad_vec, _ = self._legacy.costGradient(z)
            self._grad_cache = grad_vec
        return value

    def grad(self, g):
        if self._grad_cache is None:
            grad_vec, _ = self._legacy.costGradient(self._legacy.z)
        else:
            grad_vec = self._grad_cache
        g.zero()
        g.axpy(1.0, grad_vec)
        norm = np.sqrt(g.inner(g))
        self._grad_cache = None
        return norm

    def hessian(self, zhat, Hzhat):
        Hz = self._legacy.costHessian(self._legacy.z, zhat)
        Hzhat.zero()
        Hzhat.axpy(1.0, Hz)


__all__ = ["TaylorLinearControlCostFunctional"]
