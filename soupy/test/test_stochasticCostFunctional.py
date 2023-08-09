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
# Software Foundation) version 3.0 dated June 2007.import unittest

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
    ControlModel, MeanVarRiskMeasureStochastic, meanVarRiskMeasureStochasticSettings, \
    RiskMeasureControlCostFunctional, \
    STATE, PARAMETER, ADJOINT, CONTROL
    
from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

class TestControlCostFunctional(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-1
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


    def costValueCheck(self, sample_size, use_penalization):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = True

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)

        def l2norm(u,m,z):
            return u**2*dl.dx + (m - dl.Constant(1.0))**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)
        risk = MeanVarRiskMeasureStochastic(model, prior)
        alpha = 2.0
        if use_penalization:
            penalty = L2Penalization(self.Vh, alpha)
        else:
            penalty = None

        z = model.generate_vector(CONTROL)
        z_dim = z.get_local().shape[0]

        costFun = RiskMeasureControlCostFunctional(risk, penalty)

        z.set_local(np.random.randn(z_dim))

        rng = hp.Random()
        c_val = costFun.cost(z, order=0, sample_size=sample_size, rng=rng)
        
        if isinstance(risk, MeanVarRiskMeasureStochastic):
            print("Stochastic approximation risk measure")
            print("Sample size = %d" %(len(risk.q_samples)))
            self.assertEqual(len(risk.q_samples), sample_size)
            print("Matches input sample size")

        rng = hp.Random()
        risk.computeComponents(z, order=0, sample_size=sample_size, rng=rng)
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
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)

        # 4. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureStochasticSettings()
        rm_settings['beta'] = 0.5

        risk = MeanVarRiskMeasureStochastic(model, prior, rm_settings)
        alpha = 2.0
        penalty = L2Penalization(self.Vh, alpha)
        costFun = RiskMeasureControlCostFunctional(risk, penalty)

        z0 = model.generate_vector(CONTROL)
        dz = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)
        g0 = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        rng = hp.Random()
        c0 = costFun.cost(z0, order=1, sample_size=sample_size, rng=rng)
        costFun.grad(g0)

        rng = hp.Random()
        c1 = costFun.cost(z1, order=0, sample_size=sample_size, rng=rng)

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

    def testRiskMeasureCostFunctional(self):
        self.costValueCheck(1, True)
        self.costValueCheck(10, True)
        self.costValueCheck(1, False)
        self.costValueCheck(10, False)

        self.finiteDifferenceCheck(1, True)
        self.finiteDifferenceCheck(10, True)
        self.finiteDifferenceCheck(1, False)
        self.finiteDifferenceCheck(10, False)


if __name__ == "__main__":
    unittest.main()

