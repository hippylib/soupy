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
import os, sys

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append('../../')
from soupy import ControlCostFunctional, PDEVariationalControlProblem, \
    VariationalControlQoI, L2Penalization, \
    ControlModel, MeanVarRiskMeasureSAA, meanVarRiskMeasureSAASettings, \
    RiskMeasureControlCostFunctional, ScipyCostWrapper, \
    STATE, PARAMETER, ADJOINT, CONTROL, \
    MultipleSerialPDEsCollective
    
from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

def qoi_for_testing(u,m,z):
    return u**2*dl.dx + dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.ds


def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)


class TestControlCostFunctional(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-4
        self.nx = 16
        self.ny = 16

        self.comm_mesh = MPI.COMM_SELF 
        self.comm_sampler = MPI.COMM_WORLD
        self.rank_sampler = self.comm_sampler.Get_rank()
        self.size_sampler = self.comm_sampler.Get_size()
        self.collective = MultipleSerialPDEsCollective(self.comm_sampler)

        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.comm_mesh, self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 2)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 1)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

        def res_form(u,m,p,z):
            return dl.exp(m)*dl.inner(dl.grad(u), dl.grad(p))*dl.dx + u**3 * p * dl.dx - z*p*dl.dx 

        bc = dl.DirichletBC(Vh_STATE, dl.Constant(1.0), u_boundary)
        bc0 = dl.DirichletBC(Vh_STATE, dl.Constant(0.0), u_boundary)
        self.pde = PDEVariationalControlProblem(self.Vh, res_form, [bc], [bc0], False)

        self.prior_gamma = 5.0
        self.prior_delta = 10.0 
        self.parameter_prior = hp.BiLaplacianPrior(self.Vh[PARAMETER], self.prior_gamma, self.prior_delta, robin_bc=True)
        self.parameter_noise = dl.Vector(self.mesh.mpi_comm())
        self.parameter_prior.init_vector(self.parameter_noise, "noise")

        self.control_prior = hp.BiLaplacianPrior(self.Vh[CONTROL], self.prior_gamma, self.prior_delta, robin_bc=True)
        self.control_noise = dl.Vector(self.mesh.mpi_comm())
        self.control_prior.init_vector(self.control_noise, "noise")

    def _sample_parameter(self, m):
        hp.parRandom.normal(1.0, self.parameter_noise)
        self.parameter_prior.sample(self.parameter_noise, m)

    def _sample_control(self, z):
        hp.parRandom.normal(1.0, self.control_noise)
        self.control_prior.sample(self.control_noise, z)
        self.collective.bcast(z, root=0)

    def _setup_control_model(self, pde, qoi_varf):
        qoi = VariationalControlQoI(self.Vh, qoi_varf)
        model = ControlModel(pde, qoi)
        return qoi, model

    def _setup_mean_var_risk_measure(self, model, prior, sample_size, beta):
        rm_settings = meanVarRiskMeasureSAASettings()
        rm_settings['sample_size'] = sample_size
        rm_settings['beta'] = beta
        risk = MeanVarRiskMeasureSAA(model, prior, rm_settings, comm_sampler=self.comm_sampler)
        return risk


    def costValueCheck(self, sample_size, use_penalization, alpha=2.0, beta=0.5):
        qoi, model = self._setup_control_model(self.pde, qoi_for_testing)
        risk = self._setup_mean_var_risk_measure(model, self.parameter_prior, sample_size, beta)

        if use_penalization:
            penalty = L2Penalization(self.Vh, alpha)
        else:
            penalty = None

        z = model.generate_vector(CONTROL)
        self._sample_control(z)

        cost_functional = RiskMeasureControlCostFunctional(risk, penalty)
        scipy_wrapper = ScipyCostWrapper(cost_functional, verbose=False)

        cost_value = cost_functional.cost(z, order=0)
        func = scipy_wrapper.function()
        cost_scipy = func(z.get_local())
        cost_error = cost_value - cost_scipy 


        print("---------- Check cost values are equal ----------")
        print("Rank %d Cost functional: %g" %(self.rank_sampler, cost_value))
        print("Rank %d Cost from scipy wrapper: %g" %(self.rank_sampler, cost_scipy))
        sys.stdout.flush()
        if abs(cost_value) > 0:
            self.assertTrue(abs(cost_error/cost_value) < self.reltol)
        else:
            self.assertTrue(cost_error < self.reltol)
        
        # check that all processes have the same cost 
        cost_allprocs = self.comm_sampler.allgather(cost_value)
        self.assertTrue(np.allclose(cost_allprocs, cost_allprocs[0]))


    def testCostValue(self):
        sample_size = 10
        use_penalization = False
        self.costValueCheck(sample_size, use_penalization)

        use_penalization = True
        self.costValueCheck(sample_size, use_penalization)

    def finiteDifferenceCheck(self, sample_size, use_penalization, alpha=2.0, beta=0.5):
        print("-" * 80)
        qoi, model = self._setup_control_model(self.pde, qoi_for_testing)
        risk = self._setup_mean_var_risk_measure(model, self.parameter_prior, sample_size, beta)

        if use_penalization:
            penalty = L2Penalization(self.Vh, alpha)
        else:
            penalty = None

        cost_functional = RiskMeasureControlCostFunctional(risk, penalty)
        scipy_wrapper = ScipyCostWrapper(cost_functional, verbose=False)

        cost_fun = scipy_wrapper.function()
        grad_fun = scipy_wrapper.jac()
        hess_fun = scipy_wrapper.hessp()


        g = cost_functional.generate_vector(CONTROL)

        z0 = cost_functional.generate_vector(CONTROL)
        dz = cost_functional.generate_vector(CONTROL)
        self._sample_control(z0)
        self._sample_control(dz)

        z0_np = z0.get_local()
        dz_np = dz.get_local()
        z1_np = z0_np + self.delta * dz_np
        
        c0 = cost_fun(z0_np)
        g0_np = grad_fun(z0_np)
        Hdz_ad = hess_fun(z0_np, dz_np)

        c1 = cost_fun(z1_np)
        g1_np = grad_fun(z1_np)

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = np.inner(g0_np, dz_np)
        print("Initial cost: ", c0)
        print("New cost: ", c1)
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

        Hdz_fd = (g1_np - g0_np)/self.delta
        print("Finite difference Hessian action norm: %g" %(np.linalg.norm(Hdz_fd)))
        print("Adjoint Hessian action norm: %g" %(np.linalg.norm(Hdz_ad)))
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Error norm: %g" %(err_hess))
        self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testFiniteDifference(self):
        sample_size = 10
        use_penalization = False
        self.finiteDifferenceCheck(sample_size, use_penalization)

        use_penalization = True
        self.finiteDifferenceCheck(sample_size, use_penalization)

    
    def testScipyOptimize(self):
        """
        Test that the function handles can be used with scipy.optimize 
        """
        try:
            from scipy.optimize import minimize
            has_scipy = True
        except ImportError:
            has_scipy = False 

        if has_scipy: 
            sample_size = 10
            beta = 0.5
            alpha = 2.0
            qoi, model = self._setup_control_model(self.pde, qoi_for_testing)
            risk = self._setup_mean_var_risk_measure(model, self.parameter_prior, sample_size, beta)
            penalty = L2Penalization(self.Vh, alpha)
            z = model.generate_vector(CONTROL)
            self._sample_control(z)
            z0 = z.get_local()
            cost_functional = RiskMeasureControlCostFunctional(risk, penalty)
            scipy_wrapper = ScipyCostWrapper(cost_functional, verbose=False)

            cost_fun= scipy_wrapper.function()
            grad_fun = scipy_wrapper.jac()
            hess_fun = scipy_wrapper.hessp()
            minimize(cost_fun, z0, method="Newton-CG", jac=grad_fun, hessp=hess_fun, options={"maxiter" : 10, "disp" : True})

if __name__ == "__main__":
    unittest.main()

