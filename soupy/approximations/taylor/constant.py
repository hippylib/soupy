"""Zeroth-order Taylor approximation for risk-aware control.

When MC correction is enabled, this becomes equivalent to Sample Average
Approximation (SAA), as the constant Taylor term cancels out in the correction.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import dolfin as dl
from hippylib import ParameterList, Random

from ...modeling.controlCostFunctional import (
    ControlCostFunctional,
    DeterministicControlCostFunctional,
)
from ...modeling.variables import CONTROL, STATE, PARAMETER, ADJOINT
from .settings import taylor_constant_settings


def _merge_settings(
    user_settings: Optional[Union[dict, ParameterList]],
) -> ParameterList:
    """Return a ParameterList containing defaults overridden by user input."""
    return taylor_constant_settings(user_settings)


class TaylorConstantControlCostFunctional(ControlCostFunctional):
    """Deterministic surrogate obtained from the zeroth-order Taylor expansion.

    With MC correction enabled, this computes:
        E[Q] ≈ (1/N) Σ Q(m_i)  (SAA estimator)
        Var[Q] ≈ (1/N) Σ Q(m_i)² - ((1/N) Σ Q(m_i))²

    The Taylor constant term Q(m̄) serves as a (trivial) control variate.
    """

    def __init__(
        self,
        control_model,
        prior,
        penalization=None,
        settings: Optional[Union[dict, ParameterList]] = None,
    ):
        self.settings = _merge_settings(settings)
        self._deterministic = DeterministicControlCostFunctional(
            control_model, prior, penalization
        )
        self.prior = prior
        self.pde = control_model.problem
        self.qoi = control_model.qoi

        # Settings
        self.correction = self.settings["correction"]
        self.N_mc = self.settings["N_mc"]
        self.beta = self.settings["beta"]
        try:
            self.verbose = self.settings["verbose"]
        except (KeyError, ValueError):
            self.verbose = False

        # Book-keeping arrays
        self.lin_mean = 0.0
        self.lin_var = 0.0
        self.lin_diff_mean = np.zeros(self.N_mc) if self.N_mc > 0 else []
        self.lin_diff_var = np.zeros(self.N_mc) if self.N_mc > 0 else []
        self.Q_mc = np.zeros(self.N_mc) if self.N_mc > 0 else []

        self.last_objective = 0.0
        self.last_penalization = 0.0

        # MPI setup
        self.mpi_rank = dl.MPI.rank(self.pde.Vh[STATE].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(self.pde.Vh[STATE].mesh().mpi_comm())

        # MC samples setup
        if self.correction and self.N_mc > 0:
            self.m_mc = []
            self.x_mc = []
            randomGen = Random(myid=0, nproc=self.mpi_size)
            for _ in range(self.N_mc):
                noise = dl.Vector()
                prior.init_vector(noise, "noise")
                randomGen.normal(1.0, noise)

                sample = dl.Vector()
                prior.init_vector(sample, 1)
                prior.sample(noise, sample, add_mean=True)  # Full sample m_i = m̄ + δm
                self.m_mc.append(sample)
                self.x_mc.append(self.pde.generate_state())

    @property
    def model(self):
        return self._deterministic.model

    def generate_vector(self, component="ALL"):
        return self._deterministic.generate_vector(component)

    def cost(self, z, order=0):
        # Compute deterministic cost at prior mean
        value = self._deterministic.cost(z, order=order)

        # Capture diagnostic values
        Q0 = self.model.cost(self._deterministic.x)
        self.lin_mean = Q0
        self.lin_var = Q0 ** 2  # E[Q²] for constant Taylor (no variance term)
        self.last_objective = Q0

        if self._deterministic.penalization is None:
            self.last_penalization = 0.0
        else:
            self.last_penalization = self._deterministic.penalization.cost(
                self._deterministic.z
            )

        # Monte Carlo correction (makes this equivalent to SAA)
        mean_diff = 0.0
        var_diff = 0.0

        if self.correction and self.N_mc > 0:
            z_vec = self._deterministic.z

            for i in range(self.N_mc):
                m_i = self.m_mc[i]

                # Solve forward problem at m_i
                x_all = [self.x_mc[i], m_i, self.pde.generate_state(), z_vec]
                self.pde.solveFwd(self.x_mc[i], x_all)

                # Compute true QoI value Q(m_i)
                Q_i = self.qoi.cost([self.x_mc[i], m_i, None, z_vec])
                self.Q_mc[i] = Q_i

                # Taylor constant approximation: Q_taylor = Q0 = Q(m̄)
                Q_taylor = Q0

                # Store corrections
                self.lin_diff_mean[i] = Q_i - Q_taylor
                self.lin_diff_var[i] = Q_i ** 2 - Q_taylor ** 2

            mean_diff = np.mean(self.lin_diff_mean)
            var_diff = np.mean(self.lin_diff_var)

        # Corrected statistics
        corrected_mean = self.lin_mean + mean_diff
        corrected_var = self.lin_var + var_diff

        # Risk-averse cost: E[Q] + β * Var[Q]
        objective = corrected_mean + self.beta * (corrected_var - corrected_mean ** 2)

        if self.verbose and self.mpi_rank == 0:
            print("  [Constant Taylor] Q(m̄)={:.3e}".format(Q0))
            if self.correction and self.N_mc > 0:
                saa_mean = np.mean(self.Q_mc)
                saa_var = np.var(self.Q_mc)
                print("                    SAA mean={:.3e}, SAA var={:.3e}".format(
                    saa_mean, saa_var))

        return objective + self.last_penalization

    def grad(self, g):
        return self._deterministic.grad(g)

    def hessian(self, zhat, Hzhat):
        return self._deterministic.hessian(zhat, Hzhat)

    def init_parameter(self, m):
        self.model.init_parameter(m)

    def init_control(self, z):
        if self._deterministic.penalization is not None:
            self._deterministic.penalization.init_vector(z)
        else:
            self.model.init_control(z)

    @property
    def control_dimension(self):
        return self.model.problem.Vh[CONTROL].dim()


__all__ = ["TaylorConstantControlCostFunctional"]
