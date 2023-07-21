# Copyright (c) 2023, The University of Texas at Austin 
# & Georgia Institute of Technology
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the SOUPy package. For more information see
# https://github.com/hippylib/soupy/
#
# SOUPy is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 1991.import unittest

import unittest

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import ControlCostFunctional, PDEVariationalControlProblem, \
    VariationalControlQoI, L2Penalization, \
    ControlModel, MeanVarRiskMeasureSAA, meanVarRiskMeasureSAASettings, \
    RiskMeasureControlCostFunctional, \
    STATE, PARAMETER, ADJOINT, CONTROL
    
from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

def l2_norm(u,m,z):
    return u**2*dl.dx + (m - dl.Constant(1.0))**2*dl.dx

def qoi_for_testing(u,m,z):
    return u**2*dl.dx + dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.ds

class TestControlCostFunctional(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-3
        self.n_wells_per_side = 3
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2

        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    def _setup_pde_and_distributions(self, is_fwd_linear):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
        return pde, prior, control_dist

    def _setup_control_model(self, pde, qoi_varf):
        qoi = VariationalControlQoI(self.mesh, self.Vh, qoi_varf)
        model = ControlModel(pde, qoi)
        return qoi, model

    def _setup_mean_var_risk_measure(self, model, prior, sample_size, beta):
        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        risk = MeanVarRiskMeasureSAA(model, prior, rm_settings)
        return risk

    def costValueCheck(self, sample_size, use_penalization):
        IS_FWD_LINEAR = True
        ALPHA = 2.0 
        BETA = 0.0

        pde, prior, control_dist = self._setup_pde_and_distributions(IS_FWD_LINEAR)
        qoi, model = self._setup_control_model(pde, l2_norm)
        risk = self._setup_mean_var_risk_measure(model, prior, sample_size, BETA)

        if use_penalization:
            penalty = L2Penalization(self.Vh, ALPHA)
        else:
            penalty = None

        z = model.generate_vector(CONTROL)
        control_dist.sample(z)

        costFun = RiskMeasureControlCostFunctional(risk, penalty)
        c_val = costFun.cost(z, order=0)

        risk.computeComponents(z, order=0)
        r_val = risk.cost()

        if use_penalization:
            p_val = penalty.cost(z)
        else:
            p_val = 0

        c_total = r_val + p_val
        print("Cost evaluation: %g" %(c_val))
        print("Risk: %g" %(r_val))
        print("Penalization: %g" %(p_val))
        print("Combined: %g" %(r_val + p_val))

        cost_err = abs(c_val - c_total)

        if abs(c_total) > 0:
            self.assertTrue(abs(cost_err/c_val) < 1e-3)


    def finiteDifferenceCheck(self, sample_size, is_fwd_linear=True):
        print("-" * 80)
        print("Finite difference check with %d samples" %(sample_size))
        if is_fwd_linear:
            print("Linear problem")
        else:
            print("Nonlinear problem")
        
        ALPHA = 0.0
        BETA = 0.5
        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)
        risk = self._setup_mean_var_risk_measure(model, prior, sample_size, BETA)

        penalty = L2Penalization(self.Vh, ALPHA)
        costFun = RiskMeasureControlCostFunctional(risk, penalty)

        z0 = model.generate_vector(CONTROL)
        dz = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)
        g0 = model.generate_vector(CONTROL)
        g1 = model.generate_vector(CONTROL)
        Hdz = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        c0 = costFun.cost(z0, order=2)
        costFun.costGrad(z0, g0)
        costFun.costHessian(z0, dz, Hdz)

        c1 = costFun.cost(z1, order=1)
        costFun.costGrad(z1, g1)
        costFun.costGrad(z1, g1)

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

        Hdz_fd = (g1.get_local() - g0.get_local())/self.delta
        Hdz_ad = Hdz.get_local()
        print("Finite difference Hessian action \n", Hdz_fd)
        print("Adjoint Hessian action \n", Hdz_ad)
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Norm error: %g" %(err_hess))
        self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testCostValueLinearPDE(self):
        is_fwd_linear = True
        n_sample = 1
        self.costValueCheck(n_sample, is_fwd_linear)
        n_sample = 10
        self.costValueCheck(n_sample, is_fwd_linear)

    def testCostValueNonlinearPDE(self):
        is_fwd_linear = True
        n_sample = 1
        self.costValueCheck(n_sample, is_fwd_linear)
        n_sample = 10
        self.costValueCheck(n_sample, is_fwd_linear)

    def testFiniteDifferenceLinearPDE(self):
        is_fwd_linear = True
        n_sample = 1
        self.finiteDifferenceCheck(n_sample, is_fwd_linear)
        n_sample = 10
        self.finiteDifferenceCheck(n_sample, is_fwd_linear)

    def testFiniteDifferenceNonlinearPDE(self):
        is_fwd_linear = False
        n_sample = 1
        self.finiteDifferenceCheck(n_sample, is_fwd_linear)
        n_sample = 10
        self.finiteDifferenceCheck(n_sample, is_fwd_linear)


if __name__ == "__main__":
    unittest.main()

