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
# Software Foundation) version 3.0 dated June 2007.

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
                        TransformedMeanRiskMeasureSAA, \
                        transformedMeanRiskMeasureSAASettings, \
                        MeanVarRiskMeasureSAA, \
                        meanVarRiskMeasureSAASettings, \
                        IdentityFunction, FunctionWrapper, \
                        PDEVariationalControlProblem, \
                        STATE, PARAMETER, CONTROL

from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

from mpi4py import MPI

def l2_norm(u,m,z):
    return u**2*dl.dx 

def qoi_for_testing(u,m,z):
    return u**2*dl.dx + dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.ds - dl.inner(z,z)*dl.dx


class TestTransformedMeanRiskMeasureSAA(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-4
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


    def testMeanValue(self):
        """
        Test the cost value is correct compared to a mean \
            :py:class:`TransformedMeanRiskMeasureSAA` should give same results as \
            :py:class:`MeanVarRiskMeasureSAA` when using the identity function for the \
            transformation and using zero for the variance weight 

        """
        sample_size = 20
        is_fwd_linear = False
        seed = 1 

        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)
        rm_settings = transformedMeanRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['seed'] = seed
        risk = TransformedMeanRiskMeasureSAA(model, prior, settings=rm_settings, comm_sampler=self.comm_sampler)

        rm_mean_settings = meanVarRiskMeasureSAASettings()
        rm_mean_settings['beta'] = 0.0 
        rm_mean_settings['sample_size'] = sample_size
        rm_mean_settings['seed'] = seed
        mean_risk = MeanVarRiskMeasureSAA(model, prior, settings=rm_mean_settings, comm_sampler=self.comm_sampler)

        z = risk.generate_vector(CONTROL)
        control_dist.sample(z)
    
        risk.computeComponents(z, order=0)
        mean_risk.computeComponents(z, order=0)
        transformed_mean_evaluation = risk.cost()
        true_mean_evaluation = mean_risk.cost()
        
        tol = 1e-10
        self.assertTrue(abs(transformed_mean_evaluation - true_mean_evaluation) < tol)




    def _setup_pde_and_distributions(self, is_fwd_linear):
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
        return pde, prior, control_dist

    def _setup_control_model(self, pde, qoi_varf):
        qoi = VariationalControlQoI(self.Vh, qoi_varf)
        model = ControlModel(pde, qoi)
        return qoi, model

    def _setup_transformed_mean_risk_measure(self, model, prior, sample_size):
        """
        Set up the test case for the transformed mean risk measure 
        """
        rm_settings = transformedMeanRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size

        function_value = lambda x : np.sin(x)
        grad = lambda x : np.cos(x)
        hess = lambda x : -np.sin(x)
        function = FunctionWrapper(function_value, grad, hess)
        rm_settings['inner_function'] = function
        rm_settings['outer_function'] = function

        risk = TransformedMeanRiskMeasureSAA(model, prior, rm_settings, comm_sampler=self.comm_sampler)
        return risk
        
    def testSavedSolution(self):
        """
        Test that the risk measure is storing the correct solutions when evaluating \
            at different controls
        """

        IS_FWD_LINEAR = False 
        SAMPLE_SIZE = 10

        pde, prior, control_dist = self._setup_pde_and_distributions(IS_FWD_LINEAR)
        qoi, model = self._setup_control_model(pde, l2_norm)
        risk = self._setup_transformed_mean_risk_measure(model, prior, SAMPLE_SIZE)

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


    def finiteDifferenceCheck(self, risk, control_dist):

        z0 = risk.generate_vector(CONTROL)
        dz = risk.generate_vector(CONTROL)
        z1 = risk.generate_vector(CONTROL)
        g0 = risk.generate_vector(CONTROL)
        g1 = risk.generate_vector(CONTROL)
        Hdz = risk.generate_vector(CONTROL)

        # np.random.seed(1)
        control_dist.sample(z0)
        control_dist.sample(dz)
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)

        risk.computeComponents(z0, order=1)

        c0 = risk.cost()
        risk.grad(g0)
        risk.hessian(dz, Hdz)

        rng = hp.Random()        
        risk.computeComponents(z1, order=1)

        c1 = risk.cost()
        risk.grad(g1)

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
        print("Finite difference Hessian action \n", Hdz_fd)
        print("Adjoint Hessian action \n", Hdz_ad)
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Norm error: %g" %(err_hess))
        self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testMeanFiniteDifference(self):
        """
        Tests the finite difference of the :code:`TransformedMeanRiskMeasureSAA` class \
            when using the identity function 
        """
        print(80*"-")
        print("Finite difference using mean as risk measure")
        sample_size = 20
        is_fwd_linear = False
        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)

        rm_settings = transformedMeanRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        risk = TransformedMeanRiskMeasureSAA(model, prior, settings=rm_settings, comm_sampler=self.comm_sampler)

        self.finiteDifferenceCheck(risk, control_dist)

    def testTransformedMeanFiniteDifferenceLinearProblem(self):
        """
        Tests the finite difference assuming the mean risk measure using a linear PDE 
        """
        print(80*"-")
        print("Finite difference using transformed mean as risk measure (Linear PDE)")
        sample_size = 20
        is_fwd_linear = True
        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)
        risk = self._setup_transformed_mean_risk_measure(model, prior, sample_size)
        self.finiteDifferenceCheck(risk, control_dist)


    def testTransformedMeanFiniteDifferenceNonlinearProblem(self):
        """
        Tests the finite difference assuming the mean risk measure using a nonlinear PDE 
        
        """
        print(80*"-")
        print("Finite difference using transformed mean as risk measure (Nonlinear PDE)")
        sample_size = 20
        is_fwd_linear = False
        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)
        risk = self._setup_transformed_mean_risk_measure(model, prior, sample_size)
        self.finiteDifferenceCheck(risk, control_dist)


    def checkGatherSamples(self, sample_size):
        """
        Check that gather samples is collecting the samples properly \
            and that the correct number of samples is being collected
        """

        is_fwd_linear = False 
        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, l2_norm)
        risk = self._setup_transformed_mean_risk_measure(model, prior, sample_size)

        rank_sampler = self.comm_sampler.Get_rank()
        risk.q_samples = np.zeros(risk.q_samples.shape) + rank_sampler

        q_all = risk.gather_samples()
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

