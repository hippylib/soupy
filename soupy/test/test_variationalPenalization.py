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
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, L2Penalization, VariationalPenalization, AugmentedVector

class TestPenalization(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(32, 32)

    def finiteDifferenceGradient(self, penalty, z, dz, delta=1e-5, grad_tol=1e-3):
        """
        Finite difference check in direction dz 
        """
        grad = dl.Vector()
        penalty.init_vector(grad)
        if isinstance(z, AugmentedVector):
            grad = AugmentedVector(grad)

        p0 = penalty.cost(z)
        penalty.grad(z, grad)

        z.axpy(delta, dz)
        p1 = penalty.cost(z)

        finite_diff = (p1 - p0)/delta
        symbolic_derivative = grad.inner(dz)
        print("Finite difference derivative: %g" %(finite_diff))
        print("Implemented derivative: %g" %(symbolic_derivative))
        self.assertTrue(abs(finite_diff - symbolic_derivative) < grad_tol)

        # Make sure that for augmented vectors, the derivative component in t is always zero
        if isinstance(z, AugmentedVector):
            self.assertTrue(grad.get_scalar() < 1e-12)


    def finiteDifferenceHessian(self, penalty, z, zhat, delta=1e-5, hess_tol=1e-3):
        """
        Finite difference hessian check in direction dz 
        """
        grad0 = dl.Vector()
        grad1 = dl.Vector()
        Hzhat = dl.Vector()

        penalty.init_vector(grad0)
        penalty.init_vector(grad1)
        penalty.init_vector(Hzhat)

        if isinstance(z, AugmentedVector):
            grad0 = AugmentedVector(grad0)
            grad1 = AugmentedVector(grad1)
            Hzhat = AugmentedVector(Hzhat)

        penalty.grad(z, grad0)
        penalty.hessian(z, zhat, Hzhat)

        z.axpy(delta, zhat)
        penalty.grad(z, grad1)

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


    def compareFunctionValue(self, penalty, z, p_true, tol=1e-3):
        p = penalty.cost(z)
        p_err = abs(p - p_true)
        print("Evaluation: %g" %(p))
        print("Truth: %g" %(p_true))
        self.assertTrue(p_err < tol)


    def testL2Penalization(self):
        """
        Compare the VariationalPenalization against the L2Penalization class \
            Tests value for both :py:class:`dolfin.Vector` and \
            :py:class:`soupy.AugmentedVector` as inputs

        """

        print("Test function L2 Penalization")
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 2)
        Vh = [None, None, None, Vh_CONTROL]

        alpha = 2.0
        n_test = 3
        penalty_L2 = L2Penalization(Vh, alpha)

        def l2_varf(z):
            return dl.Constant(alpha) * z**2 * dl.dx 

        variational_l2 = VariationalPenalization(Vh, l2_varf)

        # Define the points 
        z_fun = dl.Function(Vh_CONTROL)
        z = z_fun.vector()
        dz_fun = dl.Function(Vh_CONTROL)
        dz = dz_fun.vector()

        for i in range(n_test):
            hp.parRandom.normal(1.0, z)
            p_true = penalty_L2.cost(z)
            self.compareFunctionValue(variational_l2, z, p_true, tol=1e-8)

            # Also check if input is an augmented vector 
            zt = AugmentedVector(z)
            self.compareFunctionValue(variational_l2, zt, p_true, tol=1e-8)



    def testFiniteDifference(self):
        print("Test finite difference using total variation")
        Vh_CONTROL = dl.FunctionSpace(self.mesh, "CG", 2)
        Vh = [None, None, None, Vh_CONTROL]

        penalty_weight = 2.0
        tv_smoothing = 1e-3
        delta = 1e-5
        tol = 1e-3

        def total_variation(z):
            return dl.Constant(penalty_weight) * dl.sqrt(dl.inner(dl.grad(z), dl.grad(z)) + dl.Constant(tv_smoothing))*dl.dx 
        
        tv_penalty = VariationalPenalization(Vh, total_variation)

        # Define the points 
        z_fun = dl.Function(Vh_CONTROL)
        z = z_fun.vector()
        dz_fun = dl.Function(Vh_CONTROL)
        dz = dz_fun.vector()

        z_expression = dl.Expression("cos(k*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", 
            k=1, degree=2)
        dz_expression = dl.Expression("cos(k*x[0])*x[1] + exp(-pow(x[0]-0.5, 2) - pow(x[1]-0.5, 2))", 
            k=2.88, degree=5)

        n_test = 3
        # check for different functions 
        for i in range(n_test):
            z_expression.k = i + 1
            dz_expression.k = i + 2.88
            z_fun.interpolate(z_expression)
            dz_fun.interpolate(dz_expression)

            self.finiteDifferenceGradient(tv_penalty, z, dz, delta=delta, grad_tol=tol)
            self.finiteDifferenceHessian(tv_penalty, z, dz, delta=delta, hess_tol=tol)

            self.finiteDifferenceGradient(tv_penalty, AugmentedVector(z), AugmentedVector(dz), delta=delta, grad_tol=tol)
            self.finiteDifferenceHessian(tv_penalty, AugmentedVector(z), AugmentedVector(dz), delta=delta, hess_tol=tol)

if __name__ == "__main__":
    unittest.main()


