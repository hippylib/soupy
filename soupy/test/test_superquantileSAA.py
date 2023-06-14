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
# Software Foundation) version 3.0 dated June 1991.

import unittest
import dolfin as dl
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats
import math

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import VariationalControlQoI, ControlModel, \
                        superquantileRiskMeasureSAASettings, SuperquantileRiskMeasureSAA_MPI, \
                        STATE, PARAMETER, ADJOINT, CONTROL

from poissonControlProblem import poisson_control_settings, setupPDEProblem


def standardNormalSuperquantile(beta):
    quantile = scipy.stats.norm.ppf(beta)
    return np.exp(-quantile**2/2)/(1-beta)/np.sqrt(2*math.pi)


class TestSuperquantileSAA(unittest.TestCase):
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


    def testSavedSolution(self):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False
        
        # 2. Setting up problem
        pde, prior, control_dist = setupPDEProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = 5
        rm_settings['beta'] = 0.5
        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

        zt0 = risk.generate_vector(CONTROL)
        zt1 = risk.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(zt0.get_vector())
        control_dist.sample(zt1.get_vector())
        zt0.set_scalar(np.random.randn())
        zt1.set_scalar(np.random.randn())

        # zt0.set_local(np.random.randn(len(zt0.get_local())))
        # zt1.set_local(np.random.randn(len(zt1.get_local())))
        
        print("Test if correct solution and adjoint solves are being stored")
        
        # initial cost
        risk.computeComponents(zt0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        c0 = risk.cost()
        
        # same place, check same 
        risk.computeComponents(zt0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        self.assertAlmostEqual(c0, risk.cost())
    
        # now with gradient 
        risk.computeComponents(zt0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)
        
        # new cost, no gradient 
        risk.computeComponents(zt1, order=0)
        c1 = risk.cost()
        self.assertFalse(risk.has_adjoint_solve)
        
        # back to old cost
        risk.computeComponents(zt0, order=0)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertFalse(risk.has_adjoint_solve)

        # new cost, with gradient
        risk.computeComponents(zt1, order=1)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # new cost, no gradient 
        risk.computeComponents(zt1, order=0)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # old cost, no gradient 
        risk.computeComponents(zt0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)




    def finiteDifferenceCheck(self, sample_size, smoothplus_type, epsilon, is_fwd_linear=True):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        # 2. Setting up problem
        pde, prior, control_dist = setupPDEProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = 0.5
        rm_settings['smoothplus_type'] = smoothplus_type
        rm_settings['epsilon'] = epsilon

        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

        zt0 = risk.generate_vector(CONTROL)
        dzt = risk.generate_vector(CONTROL)
        zt1 = risk.generate_vector(CONTROL)
        gt0 = risk.generate_vector(CONTROL)

        control_dist.sample(zt0.get_vector())
        zt0.set_scalar(np.random.randn())
        
        control_dist.sample(dzt.get_vector())
        dzt.set_scalar(np.random.randn())

        zt1.axpy(1.0, zt0)
        zt1.axpy(self.delta, dzt)

        risk.computeComponents(zt0, order=1)
        c0 = risk.cost()
        risk.costGrad(gt0)

        rng = hp.Random()        
        risk.computeComponents(zt1, order=0, sample_size=sample_size, rng=rng)
        c1 = risk.cost()

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = gt0.inner(dzt)
        print(40*"-")
        if is_fwd_linear:
            print("Linear, %d Samples, Smooth plus type: %s" %(sample_size, smoothplus_type))
        else:
            print("Nonlinear, %d Samples, Smooth plus type: %s" %(sample_size, smoothplus_type))
        
        print("Q: ", risk.q_samples)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        print(40*"-")
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)


    def testFiniteDifference(self):
        sample_sizes = [1, 10]
        linearities = [True, False]
        smoothpluses = ["softplus", "quartic"]
        epsilons = [1e-1, 1e-4]
        
        for sample_size in sample_sizes:
            for linearity in linearities:
                for epsilon, smoothplus in zip(epsilons, smoothpluses):
                    self.finiteDifferenceCheck(sample_size, smoothplus, epsilon, linearity)

    def computeSuperquantileValue(self, sample_size, beta):
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        # 2. Setting up problem
        pde, prior, control_dist = setupPDEProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)
        np.random.seed(1)
        risk.q_samples = np.random.randn(len(risk.q_samples))
        sq = risk.superquantile()
        return sq
    

    def testSuperquantileValue(self):
        sample_size = 10000
        beta = 0.2
        sq_normal = standardNormalSuperquantile(beta)
        sq = self.computeSuperquantileValue(sample_size, beta)
        print("Computed superquantile: ", sq)
        print("Analytic superquantile: ", sq_normal)
        tol = 1e-2
        self.assertTrue(np.abs(sq_normal - sq) < tol)
        
    
    def testCostValue(self):
        beta = 0.95
        sample_size = 100
        smoothplus_types = ["softplus", "quartic"]
        epsilons = [1e-3, 1e-4]
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        tol = 5e-2
        def l2norm(u,m,z):
            return u**2*dl.dx

        for epsilon, smoothplus_type in zip(epsilons, smoothplus_types):
            rm_settings = superquantileRiskMeasureSAASettings()
            rm_settings['sample_size'] = sample_size
            rm_settings['beta'] = beta
            rm_settings['smoothplus_type'] = smoothplus_type
            rm_settings['epsilon'] = epsilon

            pde, prior, control_dist = setupPDEProblem(self.Vh, settings)
            qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
            model = ControlModel(pde, qoi)
            risk = SuperquantileRiskMeasureSAA_MPI(model, prior, rm_settings)

            zt = risk.generate_vector(CONTROL)
            z = zt.get_vector()
            control_dist.sample(z)
            # z.set_local(np.random.randn(len(z.get_local())))

            risk.computeComponents(zt, order=0)
            quantile = np.quantile(risk.q_samples, beta)
            zt.set_scalar(quantile)

            risk.computeComponents(zt, order=0)
            sq_by_cost = risk.cost()
            sq_by_samples = risk.superquantile()

            print("Superquantile from cost: ", sq_by_cost)
            print("Superquantile from samples: ", sq_by_samples)
            err = np.abs(sq_by_cost - sq_by_samples)
            self.assertTrue(err/np.abs(sq_by_samples) < tol)





if __name__ == "__main__":
    unittest.main()
