"""First-order (linear) Taylor approximation with CVaR risk measure.

For linear Taylor approximation, Q(m) ≈ Q₀ + g^T(m - m̄) where m ~ N(m̄, C).
This gives Q ~ N(μ, σ²) with μ = Q₀ and σ² = g^T C g.

The CVaR (Conditional Value at Risk) has an analytical formula for Gaussian:
    CVaR_β[Q] = μ + σ * φ(Φ⁻¹(β)) / (1-β)

where φ is the standard normal PDF and Φ is the standard normal CDF.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import dolfin as dl
import numpy as np
from hippylib import Random, vector2Function
from scipy.stats import norm

from ...modeling.controlCostFunctional import ControlCostFunctional
from ...modeling.variables import ADJOINT, CONTROL, PARAMETER, STATE
from .settings import taylor_linear_cvar_settings


OPTIMIZATION = CONTROL  # Legacy alias


def gaussian_cvar(mean, std, beta):
    """Compute CVaR for Gaussian distribution.

    CVaR_β[Q] = μ + σ * φ(Φ⁻¹(β)) / (1-β)

    Args:
        mean: Mean of the Gaussian distribution
        std: Standard deviation of the Gaussian distribution
        beta: Risk level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value
    """
    if std < 1e-14:
        return mean
    quantile = norm.ppf(beta)
    pdf_at_quantile = norm.pdf(quantile)
    return mean + std * pdf_at_quantile / (1 - beta)


def gaussian_var(mean, std, beta):
    """Compute VaR/quantile for a Gaussian distribution."""
    if std < 1e-14:
        return mean
    return mean + std * norm.ppf(beta)


def gaussian_cvar_grad_mean(beta):
    """Gradient of Gaussian CVaR with respect to mean.

    d(CVaR)/d(μ) = 1
    """
    return 1.0


def gaussian_cvar_grad_std(std, beta):
    """Gradient of Gaussian CVaR with respect to standard deviation.

    d(CVaR)/d(σ) = φ(Φ⁻¹(β)) / (1-β)

    Args:
        std: Standard deviation (unused but kept for API consistency)
        beta: Risk level

    Returns:
        Gradient of CVaR with respect to std
    """
    quantile = norm.ppf(beta)
    pdf_at_quantile = norm.pdf(quantile)
    return pdf_at_quantile / (1 - beta)


class _TaylorLinearCVaRLegacy:
    """Implementation of linear Taylor approximation with CVaR risk measure."""

    def __init__(self, settings, Vh, pde, qoi, prior, penalization, tol=1e-9):
        self.settings = settings
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
        self.dmq = pde.generate_parameter()

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0

        self.beta = settings["beta"]  # CVaR risk level
        self.correction = settings["correction"]
        self.N_mc = settings["N_mc"]
        self.epsilon = settings["epsilon"]

        try:
            self.verbose = settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        # Taylor statistics
        self.lin_mean = 0.0  # Q₀
        self.lin_var = 0.0  # σ² = dmq · Cdmq
        self.lin_std = 0.0  # σ

        # CVaR value
        self.cvar = 0.0
        self.var = 0.0

        # MC correction storage
        self.lin_diff_cvar = np.zeros(self.N_mc) if self.N_mc > 0 else np.array([])
        self.Q_mc = np.zeros(self.N_mc) if self.N_mc > 0 else np.array([])

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()
        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())

        if self.mpi_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        # Setup MC samples for correction
        if self.correction and self.N_mc > 0:
            self.m_mc = []
            self.x_mc = []
            self.y_mc = []
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

    def _std_gradient_scale(self):
        """Return k/sigma.

        For J_lin = mu + k * sigma, sigma = <C dmq, dmq>^(1/2), the chain rule gives
        dJ/dz = dmu/dz + (k / sigma) * <d(dmq)/dz, C dmq>.
        """
        if self.lin_std <= 1e-14:
            return 0.0
        return gaussian_cvar_grad_std(self.lin_std, self.beta) / self.lin_std

    def solve_xstar(self):
        """Solve the incremental forward problem from the CVaR variance term."""
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        y_test = dl.TestFunction(self.Vh[ADJOINT])
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        xstarrhs = self.pde.generate_state()
        scale = self._std_gradient_scale()

        if abs(scale) > 0.0:
            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmyrCdmq = dl.assemble(dl.derivative(dmrCdmq, y_fun, y_test))
            [bc.apply(dmyrCdmq) for bc in self.pde.bc0]
            xstarrhs.axpy(scale, dmyrCdmq)

        xstar = self.pde.generate_state()
        self.pde.solveIncremental(xstar, -xstarrhs, False)
        self.xstar = xstar

    def solve_ystar(self):
        """Solve the incremental adjoint problem for the linear CVaR gradient."""
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        x_test = dl.TestFunction(self.Vh[STATE])
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        ystarrhs = self.pde.generate_state()
        scale = self._std_gradient_scale()

        dxq = self.pde.generate_state()
        self.qoi.grad(STATE, self.x_all, dxq)
        [bc.apply(dxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxq)

        if abs(scale) > 0.0:
            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmxrCdmq = dl.assemble(dl.derivative(dmrCdmq, x_fun, x_test))
            [bc.apply(dmxrCdmq) for bc in self.pde.bc0]
            ystarrhs.axpy(scale, dmxrCdmq)

        xstar_fun = vector2Function(self.xstar, self.Vh[STATE])
        dxr = dl.derivative(f_form, x_fun, xstar_fun)
        dxxr = dl.assemble(dl.derivative(dxr, x_fun, x_test))
        [bc.apply(dxxr) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxr)

        dxxq = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, self.xstar, dxxq)
        [bc.apply(dxxq) for bc in self.pde.bc0]
        ystarrhs.axpy(1.0, dxxq)

        ystar = self.pde.generate_state()
        self.pde.solveIncremental(ystar, -ystarrhs, True)
        self.ystar = ystar

    def objective(self):
        """Compute the CVaR objective using linear Taylor approximation."""
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

        # Compute mean and variance from linear Taylor
        self.lin_mean = Q_0
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        m_test = dl.TestFunction(self.Vh[PARAMETER])
        self.dmq = dl.assemble(dl.derivative(f_form, m_fun, m_test))
        self.prior.Rsolver.solve(self.Cdmq, self.dmq)
        self.lin_var = self.dmq.inner(self.Cdmq)
        self.lin_std = np.sqrt(max(0.0, self.lin_var))

        # Compute analytical VaR and CVaR for Gaussian
        self.var = gaussian_var(self.lin_mean, self.lin_std, self.beta)
        # Compute analytical CVaR for Gaussian
        self.cvar = gaussian_cvar(self.lin_mean, self.lin_std, self.beta)

        # MC correction if enabled
        cvar_correction = 0.0
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

                # Linear Taylor approximation at this sample
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.assemble(dl.derivative(f_form, m_fun, m_hat))
                Q_taylor_i = Q_0 + dmr

                # Store difference for CVaR correction
                self.lin_diff_cvar[i] = Q_i - Q_taylor_i

            # Use control variate: CVaR[Q] ≈ CVaR_Taylor + E[Q - Q_Taylor]
            # This is a simplified correction; a more accurate one would be:
            # CVaR[Q] = CVaR[Q_Taylor] + (CVaR[Q] - CVaR[Q_Taylor])
            # estimated by SAA on the correction term
            cvar_correction = np.mean(self.lin_diff_cvar)

        cost = self.cvar + cvar_correction

        if self.verbose and self.mpi_rank == 0:
            print(f"  [Linear Taylor CVaR] mean={self.lin_mean:.3e}, std={self.lin_std:.3e}, CVaR={self.cvar:.3e}")
            if self.correction and self.N_mc > 0:
                print(f"                       correction={cvar_correction:.3e}")

        return cost

    def costValue(self, z):
        """Evaluate cost at control z."""
        self.func_ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
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
            print(f"  [Cost eval #{self.func_ncalls}] cost={cost:.6e}, obj={objective:.3e}, pen={penalization:.3e}")

        return cost

    def costGradient(self, z):
        """Compute gradient of cost at control z."""
        self.grad_ncalls += 1

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

        dzq = self.pde.generate_control()
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        f_form = self.pde.varf_handler(x_fun, m_fun, y_fun, z_fun)
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])
        scale = self._std_gradient_scale()

        self.solve_xstar()
        self.solve_ystar()

        if abs(scale) > 0.0:
            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmzrCdmq = dl.assemble(dl.derivative(dmrCdmq, z_fun, z_test))
            dzq.axpy(scale, dmzrCdmq)

        ystar_fun = vector2Function(self.ystar, self.Vh[ADJOINT])
        dyr = dl.derivative(f_form, y_fun, ystar_fun)
        dyzr = dl.assemble(dl.derivative(dyr, z_fun, z_test))
        dzq.axpy(1.0, dyzr)

        xstar_fun = vector2Function(self.xstar, self.Vh[STATE])
        dxr = dl.derivative(f_form, x_fun, xstar_fun)
        dxzr = dl.assemble(dl.derivative(dxr, z_fun, z_test))
        dzq.axpy(1.0, dxzr)

        dz = self.pde.generate_control()
        dz.axpy(1.0, dzq)

        if self.penalization is not None:
            dzp = self.pde.generate_control()
            self.penalization.grad(self.z, dzp)
            dz.axpy(1.0, dzp)

        self.tgrad = time.time() - tgrad

        return dz, np.sqrt(dz.inner(dz))

    def costHessian(self, z, z_dir, FD=False):
        """Compute Hessian-vector product."""
        self.hess_ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)
        if isinstance(z_dir, np.ndarray):
            idx = self.z_dir.local_range()
            self.z_dir.set_local(z_dir[idx[0]:idx[1]])
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

        # Analytical Hessian (same structure as linear Taylor mean-variance)
        Vh = self.Vh

        z_fun = vector2Function(self.z, Vh[OPTIMIZATION])
        z_dir_fun = vector2Function(self.z_dir, Vh[OPTIMIZATION])
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

        z_test = dl.TestFunction(Vh[OPTIMIZATION])
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


class TaylorLinearCVaRControlCostFunctional(ControlCostFunctional):
    """Wrapper exposing the linear Taylor CVaR approximation via modern APIs.

    Uses analytical Gaussian CVaR formula since linear Taylor gives Q ~ N(μ, σ²).
    """

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
        self.settings = taylor_linear_cvar_settings(settings)

        self._legacy = _TaylorLinearCVaRLegacy(
            self.settings,
            self.model.problem.Vh,
            self.model.problem,
            self.model.qoi,
            prior,
            penalization,
            tol=tol,
        )

        self._grad_cache = None

    @property
    def lin_mean(self):
        """Return the Taylor mean (Q₀)."""
        return self._legacy.lin_mean

    @property
    def lin_std(self):
        """Return the Taylor standard deviation."""
        return self._legacy.lin_std

    @property
    def cvar(self):
        """Return the analytical CVaR value."""
        return self._legacy.cvar

    @property
    def var(self):
        """Return the analytical VaR/quantile value."""
        return self._legacy.var

    @property
    def Q_mc(self):
        """Return MC samples (if correction enabled)."""
        return self._legacy.Q_mc

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


__all__ = ["TaylorLinearCVaRControlCostFunctional", "gaussian_cvar", "gaussian_var"]
