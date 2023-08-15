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
                        IdentityFunction, FunctionWrapper, \
                        PDEVariationalControlProblem, \
                        STATE, PARAMETER, CONTROL


class IdentityFunctionWithCounter:
    def __init__(self):
        self.n_grad = 0 
        self.n_hess = 0 

    def function(self, x):
        return x 

    def grad(self, x):
        self.n_grad += 1
        return 1

    def hessian(self, x):
        self.n_hess += 1 
        return 0



class QuarticFunction:
    def __init__(self):
        self.n_grad = 0 
        self.n_hess = 0 

    def function(self, x):
        return x**4

    def grad(self, x):
        self.n_grad += 1 
        return 4*x**3

    def hessian(self, x):
        self.n_hess += 1 
        return 12*x**2


class TestFunctionWrapper(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.fdtol = 1e-2
        self.delta = 1e-4


    def testUsingSuppliedGradient(self):
        """
        Test that wrapper uses the supplied gradient function when given as an input
        """
        identity_function = IdentityFunctionWithCounter()
        x = 0
        my_function = FunctionWrapper(identity_function.function, grad=identity_function.grad)
        grad_call = my_function.grad(x) 
        self.assertTrue(identity_function.n_grad > 0)

    def testUsingSuppliedHessian(self):
        """
        Test that wrapper uses the supplied Hessian function when given as an input
        """
        identity_function = IdentityFunctionWithCounter()
        x = 0
        my_function = FunctionWrapper(identity_function.function, hessian=identity_function.hessian)
        hess_call = my_function.hessian(x) 
        self.assertTrue(identity_function.n_hess > 0)


    def testFiniteDifferenceGradientHessian(self):
        """
        Test the finite difference approximation for both the gradient and hessian \
            when neither are supplied
        """
        quartic = QuarticFunction() 
        x = 2.0
        delta = 1e-4
        tol = 1e-3 

        quartic_wrapper = FunctionWrapper(quartic.function, delta=delta)

        true_grad = quartic.grad(x)
        true_hess = quartic.hessian(x)

        fd_grad = quartic_wrapper.grad(x)
        fd_hess = quartic_wrapper.hessian(x)
        print("-" * 80)
        print("Quartic FD Test for gradient and hessian")
        print("True gradient: %g" %(true_grad))
        print("Approximate gradient: %g" %(fd_grad))
        print("True Hessian: %g" %(true_hess))
        print("Approximate Hessian: %g" %(fd_hess))
        
        # Check that exact gradient and hessian is only called once to get truth
        self.assertEqual(quartic.n_grad, 1)
        self.assertEqual(quartic.n_hess, 1)

        self.assertTrue(abs(true_grad - fd_grad)/(abs(true_grad)) < tol)
        self.assertTrue(abs(true_hess - fd_hess)/(abs(true_hess)) < tol)


    def testFiniteDifferenceHessian(self):
        """
        Test the finite difference approximation for the Hessian \
            when only gradient is supplied
        """
        quartic = QuarticFunction() 
        x = 2.0
        delta = 1e-4
        tol = 1e-3 

        quartic_wrapper = FunctionWrapper(quartic.function, delta=delta)
        true_hess = quartic.hessian(x)
        fd_hess = quartic_wrapper.hessian(x)
        print("-" * 80)
        print("Quartic FD Test for hessian")
        print("True Hessian: %g" %(true_hess))
        print("Approximate Hessian: %g" %(fd_hess))
        self.assertTrue(abs(true_hess - fd_hess)/(abs(true_hess)) < tol)

        # Check that exact hessian is only called once to get truth
        self.assertEqual(quartic.n_hess, 1)


if __name__ == "__main__":
    unittest.main()
