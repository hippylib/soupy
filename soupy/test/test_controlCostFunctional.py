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
from soupy import VariationalControlQoI, ControlModel, DeterministicControlCostFunctional,\
                        PDEVariationalControlProblem, \
                        STATE, PARAMETER, ADJOINT, CONTROL

from setupPoissonControlProblem import poisson_control_settings, setupPoissonPDEProblem

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestControlCostFunctional(unittest.TestCase):
    def setUp(self):
        """
        Note, the source setup makes boundary effects more pronounced
        This is important for catching BC errors 
        """
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-4
        self.n_wells_per_side = 3
        self.loc_upper = 1.0
        self.loc_lower = 0.0
        self.nx = 20
        self.ny = 20
        self.n_control = self.n_wells_per_side**2
        
        # Make spaces
        self.mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]


    def testCostValue(self):
        """
        Test the evaluation of a QoI with known value
        """
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LOC_LOWER'] = self.loc_lower
        settings['LOC_UPPER'] = self.loc_upper
        settings['LINEAR'] = True

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings) 

        def l2norm(u,m,z):
            return u**2*dl.dx + (m - dl.Constant(2.0))**2*dl.dx 

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)
        cost = DeterministicControlCostFunctional(model, prior, penalization=None)
        z = model.generate_vector(CONTROL)
        c_val = cost.cost(z)
        c_val_exact = 1/3 + 1
        print("Cost %g" %(c_val))
        print("Exact cost %g" %(c_val_exact))
        
        self.assertTrue(abs((c_val - c_val_exact)/c_val_exact) < self.reltol)
        

    def finiteDifferenceCheck(self, qoi_varf, is_fwd_linear=True):
        """
        Finite difference checks the gradient and Hessian action of the cost functional
        """
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny

        settings['LOC_LOWER'] = self.loc_lower
        settings['LOC_UPPER'] = self.loc_upper
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = is_fwd_linear

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings) 
        qoi = VariationalControlQoI(self.mesh, self.Vh, qoi_varf)
        model = ControlModel(pde, qoi)
        cost = DeterministicControlCostFunctional(model, prior, penalization=None)

        z0 = model.generate_vector(CONTROL)
        dz = model.generate_vector(CONTROL)
        z1 = model.generate_vector(CONTROL)
        Hdz = model.generate_vector(CONTROL)

        control_dist.sample(z0)
        control_dist.sample(dz)

        c0 = cost.cost(z0, order=2)
        g0 = model.generate_vector(CONTROL)
        cost.grad(g0)
        cost.hessian(dz, Hdz)
        
        z1.axpy(1.0, z0)
        z1.axpy(self.delta, dz)
        
        c1 = cost.cost(z1, order=1)
        g1 = model.generate_vector(CONTROL)
        cost.grad(g1)

        dcdz_fd = (c1 - c0)/self.delta
        dcdz_ad = g0.inner(dz)
        
        print("Finite difference derivative: %g" %(dcdz_fd))
        print("Adjoint derivative: %g" %(dcdz_ad))
        print("Derivative error: %g" %(abs((dcdz_fd - dcdz_ad)/dcdz_ad)))
        self.assertTrue(abs((dcdz_fd - dcdz_ad)/dcdz_ad) < self.fdtol)

        Hdz_fd = (g1.get_local() - g0.get_local())/self.delta
        Hdz_ad = Hdz.get_local()
        print("Finite difference Hessian action \n", Hdz_fd)
        print("Adjoint Hessian action \n", Hdz_ad)
        err_hess = np.linalg.norm(Hdz_fd - Hdz_ad)
        print("Norm error: %g" %(err_hess))
        self.assertTrue(err_hess/np.linalg.norm(Hdz_ad) < self.fdtol)

    def testFiniteDifferenceL2Norm(self):
        print("Finite difference checking the L2 norm")
        def L2_norm(u,m,z):
            return dl.exp(m)*u**2 * dl.dx 
        print("Linear PDE")
        self.finiteDifferenceCheck(L2_norm, True)
        print("Nonlinear, PDE")
        self.finiteDifferenceCheck(L2_norm, False)

    def testFiniteDifferenceH1Norm(self):
        print("Finite difference checking the H1 norm")
        def H1_norm(u,m,z):
            return dl.inner(dl.exp(m)*dl.grad(u), dl.grad(u))*dl.dx 
        print("Linear PDE")
        self.finiteDifferenceCheck(H1_norm, True)
        print("Nonlinear PDE")
        self.finiteDifferenceCheck(H1_norm, False)

    def testFiniteDifferenceBoundaryL2Norm(self):
        print("Finite difference checking the L2 norm on the boundary")
        def boundary_l2_norm(u,m,z):
            return dl.exp(m)*u**2 * dl.ds 
        print("Linear PDE")
        self.finiteDifferenceCheck(boundary_l2_norm, True)
        print("Nonlinear PDE")
        self.finiteDifferenceCheck(boundary_l2_norm, False)

    def testFiniteDifferenceBoundaryH1Norm(self):
        print("Finite difference checking the H1 norm on the boundary")
        def boundary_H1_norm(u,m,z):
            return dl.inner(dl.exp(m)*dl.grad(u), dl.grad(u))*dl.ds 
        print("Linear PDE")
        self.finiteDifferenceCheck(boundary_H1_norm, True)
        print("Nonlinear PDE")
        self.finiteDifferenceCheck(boundary_H1_norm, False)

    def testSavedSolution(self):
        """
        Test if the correct solution and adjoint solves are being stored
        and re-used 
        """
        settings = poisson_control_settings()
        settings['nx'] = self.nx
        settings['ny'] = self.ny
        settings['LOC_LOWER'] = self.loc_lower
        settings['LOC_UPPER'] = self.loc_upper
        settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        settings['LINEAR'] = False

        pde, prior, control_dist = setupPoissonPDEProblem(self.Vh, settings) 

        def l2norm(u,m,z):
            return u**2*dl.dx

        qoi = VariationalControlQoI(self.mesh, self.Vh, l2norm)
        model = ControlModel(pde, qoi)
        cost = DeterministicControlCostFunctional(model, prior, penalization=None)

        z0 = cost.generate_vector(CONTROL)
        z1 = cost.generate_vector(CONTROL)

        control_dist.sample(z0)
        control_dist.sample(z1)
        
        print("Test if correct solution and adjoint solves are being stored")
        
        # initial cost
        c0 = cost.cost(z0, order=0)
        self.assertFalse(cost.has_adjoint_solve)
        
        # same place, check same 
        c_new = cost.cost(z0, order=0)
        self.assertFalse(cost.has_adjoint_solve)
        self.assertAlmostEqual(c0, c_new)
    
        # now with gradient 
        c_new = cost.cost(z0, order=1)
        self.assertAlmostEqual(c0, c_new)
        self.assertTrue(cost.has_adjoint_solve)
        
        # new cost, no gradient 
        c1 = cost.cost(z1, order=0)
        self.assertFalse(cost.has_adjoint_solve)
        
        # back to old cost
        c_new = cost.cost(z0, order=0)
        self.assertAlmostEqual(c0, c_new)
        self.assertFalse(cost.has_adjoint_solve)

        # new cost, with gradient
        c_new = cost.cost(z1, order=1)
        self.assertTrue(cost.has_adjoint_solve)
        self.assertAlmostEqual(c1, c_new)


if __name__ == "__main__":
    unittest.main()

