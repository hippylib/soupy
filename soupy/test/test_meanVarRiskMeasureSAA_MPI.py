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

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import VariationalControlQoI, ControlModel, \
                        meanVarRiskMeasureSAASettings, \
                        MeanVarRiskMeasureSAA_MPI,\
                        PDEVariationalControlProblem, \
                        STATE, PARAMETER, CONTROL

from poissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

from mpi4py import MPI


class TestMeanVarRiskMeasureSAA_MPI(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-3
        self.n_wells_per_side = 3
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        
        # Make spaces
        self.comm_mesh = MPI.COMM_SELF
        self.comm_sampler = MPI.COMM_WORLD

        self.mesh = dl.UnitSquareMesh(self.comm_mesh, self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]


    def testCostValue(self):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = True

        pde, prior, _ = setupPoissonPDEProblem(self.Vh, settings)
        def l2norm(u,m,z):
            return u**2*dl.dx + (m - dl.Constant(1.0))**2*dl.dx 

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        risk = MeanVarRiskMeasureSAA_MPI(model, prior, comm_sampler=self.comm_sampler)
        z = model.generate_vector(CONTROL)
        c_val = risk.cost()
        print("Before computing: ", c_val)
        risk.computeComponents(z)
        c_val = risk.cost()
        print("After computing: ", c_val)
        
    def testSavedSolution(self):
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False
        
        # 2. Setup problem
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
        noise = dl.Vector(self.comm_mesh)
        prior.init_vector(noise, "noise")
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = 5
        rm_settings['beta'] = 0.5
        risk = MeanVarRiskMeasureSAA_MPI(model, prior, rm_settings, comm_sampler=self.comm_sampler)

        z0 = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(z1)
        
        print("Test if correct solution and adjoint solves are being stored")
        
        # initial cost
        risk.computeComponents(z0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        c0 = risk.cost()
        
        # same place, check same 
        risk.computeComponents(z0, order=0)
        self.assertFalse(risk.has_adjoint_solve)
        self.assertAlmostEqual(c0, risk.cost())
    
        # now with gradient 
        risk.computeComponents(z0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)
        
        # new cost, no gradient 
        risk.computeComponents(z1, order=0)
        c1 = risk.cost()
        self.assertFalse(risk.has_adjoint_solve)
        
        # back to old cost
        risk.computeComponents(z0, order=0)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertFalse(risk.has_adjoint_solve)

        # new cost, with gradient
        risk.computeComponents(z1, order=1)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # new cost, no gradient 
        risk.computeComponents(z1, order=0)
        self.assertTrue(risk.has_adjoint_solve)
        self.assertAlmostEqual(c1, risk.cost())

        # old cost, no gradient 
        risk.computeComponents(z0, order=1)
        self.assertAlmostEqual(c0, risk.cost())
        self.assertTrue(risk.has_adjoint_solve)


    def finiteDifferenceCheck(self, sample_size, is_fwd_linear=True):
        print("-" * 80)
        print("Finite difference check with %d samples" %(sample_size))
        if is_fwd_linear:
            print("Linear problem")
        else:
            print("Nonlinear problem")


        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        # 2. Setting up problem
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
        noise = dl.Vector(self.comm_sampler)
        prior.init_vector(noise, "noise")
    
        # 3. Setting up QoI, model, and risk measure
        def qoi_for_testing(u,m,z):
            return u**2*dl.dx + dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.ds

        qoi = VariationalControlQoI(self.mesh, self.Vh, qoi_for_testing)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = 0.5

        risk = MeanVarRiskMeasureSAA_MPI(model, prior, rm_settings, comm_sampler=self.comm_sampler)

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

        risk.computeComponents(z0, order=1)

        c0 = risk.cost()
        risk.costGrad(g0)
        risk.costHessian(dz, Hdz)

        rng = hp.Random()        
        risk.computeComponents(z1, order=1)

        c1 = risk.cost()
        risk.costGrad(g1)

        # Test gradients
        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

        # Test hessians
        Hdz_fd = (g1.get_local() - g0.get_local())/self.delta
        Hdz_ad = Hdz.get_local()
        print("Finite difference Hessian action ", Hdz_fd)
        print("Adjoint Hessian action ", Hdz_ad)
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Norm error: %g" %(err_hess))
        self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testFiniteDifferenceLinearProblem(self):
        is_fwd_linear = True
        n_samples = 100
        self.finiteDifferenceCheck(n_samples, is_fwd_linear)

    def testFiniteDifferenceNonlinearProblem(self):
        is_fwd_linear = False
        n_samples = 100
        self.finiteDifferenceCheck(n_samples, is_fwd_linear)

    def checkGatherSamples(self, sample_size):
        """
        Check that gather samples is collecting the samples properly \
            and that the correct number of samples is being collected
        """
        # 1. Settings for PDE
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False
        
        # 2. Setup problem
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = 0.5
        rank_sampler = self.comm_sampler.Get_rank()
        risk = MeanVarRiskMeasureSAA_MPI(model, prior, rm_settings, comm_sampler=self.comm_sampler)
        risk.q_samples = np.zeros(risk.q_samples.shape) + rank_sampler

        q_all = risk.gatherSamples()
        number_equal_to_rank = np.sum(np.isclose(q_all, rank_sampler))

        print("Testing samples are gathering correctly")
        print("Number of samples equal to rank (%d): %d" %(rank_sampler, number_equal_to_rank))
        print("Should have %g" %(risk.sample_size_allprocs[rank_sampler]))
        self.assertTrue(np.isclose(number_equal_to_rank, risk.sample_size_allprocs[rank_sampler]))
        self.assertTrue(len(q_all) == sample_size)


    def testGatherSamples(self):
        SAMPLE_SIZE_ODD = 2017
        SAMPLE_SIZE_EVEN = 720
        self.checkGatherSamples(SAMPLE_SIZE_ODD)
        self.checkGatherSamples(SAMPLE_SIZE_EVEN)


if __name__ == "__main__":
    unittest.main()

