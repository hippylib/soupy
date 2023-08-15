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

import numpy as np
import dolfin as dl

import sys, os 
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

sys.path.append('../../')
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, VariationalPenalization, AugmentedVector, \
        PenalizationControlCostFunctional

class TestPenalization(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(16, 16)
        Vh_STATE = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, "CG", 1)
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 1)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    def testCostValue(self):

        def l2_form(z):
            return z**2*dl.dx 
        
        l2_penalty = VariationalPenalization(self.Vh, l2_form)
        
        Z_VALUE = 2.0 
        TRUE_COST = 4.0 

        z0 = dl.interpolate(dl.Constant(Z_VALUE), self.Vh[CONTROL]).vector()
        zt0 = AugmentedVector(z0)
        zt0.set_scalar(np.random.randn())

        l2_cost = PenalizationControlCostFunctional(self.Vh, l2_penalty)
        l2_cost_augmented = PenalizationControlCostFunctional(self.Vh, l2_penalty, augment_control=True)

        cost_value = l2_cost.cost(z0)
        cost_value_from_augmented = l2_cost_augmented.cost(zt0)
        
        print("-" * 80)
        print("Checking cost values are consistent")
        print("True: %g" %(TRUE_COST))
        print("Cost from penalization: %g" %(cost_value))
        print("Cost from penalization with augmented vector: %g" %(cost_value_from_augmented))

        tol = 1e-12
        self.assertTrue( abs(TRUE_COST - cost_value) < tol )
        self.assertTrue( abs(TRUE_COST - cost_value_from_augmented) < tol )


    def finiteDifferenceGradient(self, cost_penalty, z, dz, delta=1e-5, grad_tol=1e-3):
        """
        Finite difference check in direction dz 
        """
        grad = cost_penalty.generate_vector(CONTROL)

        p0 = cost_penalty.cost(z, order=1)
        cost_penalty.grad(grad)

        z.axpy(delta, dz)
        p1 = cost_penalty.cost(z, order=0)

        finite_diff = (p1 - p0)/delta
        symbolic_derivative = grad.inner(dz)
        print("Finite difference derivative: %g" %(finite_diff))
        print("Implemented derivative: %g" %(symbolic_derivative))
        self.assertTrue(abs(finite_diff - symbolic_derivative) < grad_tol)

        # Make sure that for augmented vectors, the derivative component in t is always zero
        if isinstance(z, AugmentedVector):
            self.assertTrue(grad.get_scalar() < 1e-12)


    def finiteDifferenceHessian(self, cost_penalty, z, zhat, delta=1e-5, hess_tol=1e-3):
        """
        Finite difference hessian check in direction dz 
        """
        grad0 = cost_penalty.generate_vector(CONTROL)
        grad1 = cost_penalty.generate_vector(CONTROL)
        Hzhat = cost_penalty.generate_vector(CONTROL)
        
        cost_penalty.cost(z, order=2)
        cost_penalty.grad(grad0)
        cost_penalty.hessian(zhat, Hzhat)

        z.axpy(delta, zhat)
        cost_penalty.cost(z, order=1)
        cost_penalty.grad(grad1)

        Hzhat_fd = (grad1.get_local() - grad0.get_local())/delta
        hess_diff = Hzhat_fd - Hzhat.get_local()

        if np.linalg.norm(Hzhat.get_local()) > 0:
            hess_err = np.linalg.norm(hess_diff)/np.linalg.norm(Hzhat.get_local())
        else:
            hess_err = np.linalg.norm(hess_diff)

        print("Hessian difference: %g" %(hess_err))
        self.assertTrue(hess_err < hess_tol)

        # Make sure that for augmented vectors, the derivative component in t is always zero
        if isinstance(z, AugmentedVector):
            self.assertTrue(Hzhat.get_scalar() < 1e-12)


    def testFiniteDifference(self):
        print("Test finite difference using total variation")
        penalty_weight = 2.0
        tv_smoothing = 1e-3
        delta = 1e-5
        tol = 1e-3

        def total_variation(z):
            return dl.Constant(penalty_weight) * dl.sqrt(dl.inner(dl.grad(z), dl.grad(z)) + dl.Constant(tv_smoothing))*dl.dx 
        
        tv_penalty = VariationalPenalization(self.Vh, total_variation)

        # Define the points 
        z_fun = dl.Function(self.Vh[CONTROL])
        z = z_fun.vector()
        dz_fun = dl.Function(self.Vh[CONTROL])
        dz = dz_fun.vector()

        z_expression = dl.Expression("cos(k*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", 
            k=1, degree=2)
        dz_expression = dl.Expression("cos(k*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", 
            k=2.88, degree=5)

        n_test = 3
        # check for different functions 

        cost_penalty = PenalizationControlCostFunctional(self.Vh, tv_penalty)
        cost_penalty_augmented = PenalizationControlCostFunctional(self.Vh, tv_penalty, augment_control=True)

        for i in range(n_test):
            z_expression.k = i + 1
            dz_expression.k = i + 2.88
            z_fun.interpolate(z_expression)
            dz_fun.interpolate(dz_expression)
            print("-" * 80) 
            print("FD test %d, penalty cost functional" %(i))
            self.finiteDifferenceGradient(cost_penalty, z, dz, delta=delta, grad_tol=tol)
            self.finiteDifferenceHessian(cost_penalty, z, dz, delta=delta, hess_tol=tol)

            print("-" * 80) 
            print("FD test %d, augmented penalty cost functional" %(i))
            self.finiteDifferenceGradient(cost_penalty_augmented, AugmentedVector(z), AugmentedVector(dz), delta=delta, grad_tol=tol)
            self.finiteDifferenceHessian(cost_penalty_augmented, AugmentedVector(z), AugmentedVector(dz), delta=delta, hess_tol=tol)

if __name__ == "__main__":
    unittest.main()


