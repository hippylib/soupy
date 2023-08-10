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
import scipy.stats
import math

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

from mpi4py import MPI 

import os, sys
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import VariationalControlQoI, ControlModel, \
                        superquantileRiskMeasureSAASettings, SuperquantileRiskMeasureSAA, \
                        STATE, PARAMETER, ADJOINT, CONTROL, MultipleSerialPDEsCollective

from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem


def standardNormalSuperquantile(beta):
    quantile = scipy.stats.norm.ppf(beta)
    return np.exp(-quantile**2/2)/(1-beta)/np.sqrt(2*math.pi)


def sample_zt_and_bcast(control_dist, zt, collective, root=0):
    """
    Samples zt and broadcast from root 
    """
    z = zt.get_vector()

    # sample the scalar and broadcast
    # t = np.random.randn()
    # t = collective.bcast(t, root=root)
    t = -1 
    zt.set_scalar(t)

    # sample the vector and broadcast 
    control_dist.sample(z)
    collective.bcast(z)

def l2_norm(u,m,z):
    return u**2*dl.dx 

def qoi_for_testing(u,m,z):
    return u**2*dl.dx + dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.ds


class TestSuperquantileSAA(unittest.TestCase):
    """
    Testing :code:`soupy.SuperquantileRiskMeasureSAA` with MPI \
            capabilities enabled
    """
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-3
        self.n_wells_per_side = 3
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        self.comm_mesh = MPI.COMM_SELF
        self.comm_sampler = MPI.COMM_WORLD
        self.collective_sampler = MultipleSerialPDEsCollective(self.comm_sampler)

        self.rank_sampler = self.comm_sampler.Get_rank()
        self.size_sampler = self.comm_sampler.Get_rank()


        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.comm_mesh, self.nx, self.ny)
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
        qoi = VariationalControlQoI(self.Vh, qoi_varf)
        model = ControlModel(pde, qoi)
        return qoi, model


    def _setup_superquantile_risk_measure(self, model, prior, sample_size, beta, 
                smoothplus_type='quartic', epsilon=1e-4):
        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        rm_settings['smoothplus_type']
        rm_settings['epsilon']
        risk = SuperquantileRiskMeasureSAA(model, prior, rm_settings, comm_sampler=self.comm_sampler)
        return risk 

    def testSavedSolution(self):
        """
        Testing correct solutions are saved internally 
        """
        IS_FWD_LINEAR = False 
        SAMPLE_SIZE = 100
        BETA = 0.5 

        pde, prior, control_dist = self._setup_pde_and_distributions(IS_FWD_LINEAR)
        qoi, model = self._setup_control_model(pde, l2_norm)
        risk = self._setup_superquantile_risk_measure(model, prior, SAMPLE_SIZE, BETA)
    
        zt0 = risk.generate_vector(CONTROL)
        zt1 = risk.generate_vector(CONTROL)

        sample_zt_and_bcast(control_dist, zt0, self.collective_sampler)
        sample_zt_and_bcast(control_dist, zt1, self.collective_sampler)

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
        """
        Finite difference checks with gradients and hessians 
        """
        BETA = 0.5

        pde, prior, control_dist = self._setup_pde_and_distributions(is_fwd_linear)
        qoi, model = self._setup_control_model(pde, qoi_for_testing)
        risk = self._setup_superquantile_risk_measure(model, prior, sample_size, BETA,
            smoothplus_type=smoothplus_type, epsilon=epsilon)

        zt0 = risk.generate_vector(CONTROL)
        dzt = risk.generate_vector(CONTROL)
        zt1 = risk.generate_vector(CONTROL)
        gt0 = risk.generate_vector(CONTROL)
        gt1 = risk.generate_vector(CONTROL)
        Hdzt = risk.generate_vector(CONTROL)

        sample_zt_and_bcast(control_dist, zt0, self.collective_sampler)
        sample_zt_and_bcast(control_dist, dzt, self.collective_sampler)

        zt1.axpy(1.0, zt0)
        zt1.axpy(self.delta, dzt)

        risk.computeComponents(zt0, order=1)
        c0 = risk.cost()
        risk.grad(gt0)
        risk.hessian(dzt, Hdzt)

        risk.computeComponents(zt1, order=1)
        c1 = risk.cost()
        risk.grad(gt1)

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
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)


        Hdz_fd = (gt1.get_local() - gt0.get_local())/self.delta
        Hdz_ad = Hdzt.get_local()
        print("Finite difference Hessian action \n", Hdz_fd)
        print("Adjoint Hessian action \n", Hdz_ad)
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Norm error: %g" %(err_hess))

        if np.allclose(Hdz_ad, 0):
            self.assertTrue(err_hess < self.fdtol)
        else:
            norm_hess = np.linalg.norm(Hdz_ad)
            self.assertTrue(err_hess/norm_hess < self.fdtol)
        print(40*"-")

    def testFiniteDifference(self):
        """
        Test finite difference. If testing with MPI, 
        test with sample sizes greater than one only
        """
        if MPI.COMM_WORLD.Get_size() == 1:
            sample_sizes = [1, 10]
        else:
            sample_sizes = [100]

        linearities = [True, False]
        smoothpluses = ["softplus", "quartic"]
        epsilons = [1e-1, 1e-4]
        
        for sample_size in sample_sizes:
            for linearity in linearities:
                for epsilon, smoothplus in zip(epsilons, smoothpluses):
                    self.finiteDifferenceCheck(sample_size, smoothplus, epsilon, linearity)



    def computeSuperquantileValue(self, sample_size, beta):
        """
        Evaluate superquantile using :code:`soupy.SuperquantileRiskMeasureSAA.superquantile` \
                for samples drawn from a normal distribution 
        
        """
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        # 2. Setting up problem
        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings)
    
        # 3. Setting up QoI, model, and risk measure
        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.Vh, l2norm)
        model = ControlModel(pde, qoi)

        rm_settings = superquantileRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        risk = SuperquantileRiskMeasureSAA(model, prior, rm_settings, comm_sampler=self.comm_sampler)
        
        np.random.seed(1)
        # Sample all of the samples 
        q_samples_all = np.random.randn(sample_size)

        # Partition samples for each process 
        ind_samples_end = np.cumsum(risk.sample_size_allprocs)
        ind_samples_start = ind_samples_end - ind_samples_end[0]
         
        risk.q_samples = q_samples_all[ind_samples_start[self.rank_sampler] : ind_samples_end[self.rank_sampler]]
        sq = risk.superquantile()
        return sq
    
    def checkGatherSamples(self, sample_size):
        """
        Check that gather samples is collecting the samples properly \
            and that the correct number of samples is being collected
        """
        IS_FWD_LINEAR = False 
        BETA = 0.5 

        pde, prior, control_dist = self._setup_pde_and_distributions(IS_FWD_LINEAR)
        qoi, model = self._setup_control_model(pde, l2_norm)
        risk = self._setup_superquantile_risk_measure(model, prior, sample_size, BETA)

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

    def testSuperquantileValue(self):
        """
        Test the superquantile computation when samples are spread across multiple processes
        """
        sample_size = 10000
        beta = 0.2
        sq_normal = standardNormalSuperquantile(beta)
        sq = self.computeSuperquantileValue(sample_size, beta)
        print("Computed superquantile: ", sq)
        print("Analytic superquantile: ", sq_normal)
        tol = 1e-2
        self.assertTrue(np.abs(sq_normal - sq) < tol)
        
    
    def testCostValue(self):
        BETA = 0.95
        SAMPLE_SIZE = 100
        IS_FWD_LINEAR = False

        pde, prior, control_dist = self._setup_pde_and_distributions(IS_FWD_LINEAR)
        qoi, model = self._setup_control_model(pde, l2_norm)


        smoothplus_types = ["softplus", "quartic"]
        epsilons = [1e-3, 1e-4]
        settings = poisson_control_settings()
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side

        tol = 5e-2
        def l2norm(u,m,z):
            return u**2*dl.dx

        for epsilon, smoothplus_type in zip(epsilons, smoothplus_types):
            risk = self._setup_superquantile_risk_measure(model, prior, 
                SAMPLE_SIZE, BETA, smoothplus_type=smoothplus_type, epsilon=epsilon)

            zt = risk.generate_vector(CONTROL)
            sample_zt_and_bcast(control_dist, zt, self.collective_sampler)

            risk.computeComponents(zt, order=0)
            quantile = np.percentile(risk.gather_samples(), BETA * 100)
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

