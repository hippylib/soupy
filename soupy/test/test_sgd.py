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

import numpy as np
import dolfin as dl

import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

sys.path.append('../../')
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, \
    SGD, SGD_ParameterList, InnerProductEqualityConstraint

class QuadraticCost:
    def __init__(self, Vz, dim):
        self.dim = dim
        self.Vz = Vz

    def generate_vector(self, component=CONTROL):
        assert component == CONTROL
        return dl.Function(self.Vz).vector()


    def cost(self, z, order=0, sample_size=1, rng=None):
        return z.inner(z)

    def costGrad(self, z, out):
        out.zero()
        out.axpy(2.0, z)

    def costHessian(self, z, zhat, out):
        out.zero()
        out.axpy(zhat)



class TestSGD(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(10, 10)
        self.dim = 2
        self.Vz = dl.VectorFunctionSpace(self.mesh, "R", 0, dim=self.dim)
        self.cost = QuadraticCost(self.Vz, self.dim)


    def runOptimization(self, max_iter, step_size, box_bounds, other_constraint=None, use_sa=True):
        sgd_param = SGD_ParameterList()
        sgd_param["max_iter"] = max_iter
        sgd_param["alpha"] = step_size
        sgd_param["stochastic_approximation"] = use_sa
        sgd_param["print_level"] = 0

        sgd_solver = SGD(self.cost, sgd_param)

        z0 = dl.Function(self.Vz).vector()
        np.random.randn(1)
        z0.set_local(np.random.randn(self.dim))

        if box_bounds is not None:
            param_min = box_bounds[0]
            param_max = box_bounds[1]
            z0.set_local(np.maximum(z0.get_local(), param_min))
            z0.set_local(np.minimum(z0.get_local(), param_max))

        if other_constraint is not None:
            other_constraint.project(z0)

        z_opt, _ = sgd_solver.solve(z0, box_bounds, other_constraint)
        print("Optimal z: ")
        print(z_opt.get_local())
        return z_opt.get_local()


    def testOptimization(self):
        # Unconstrained quadratic case
        for use_sa, step_size in zip([True, False], [0.1, 0.5]):
            tol = 1e-3
            max_iter = 100
            box_bounds = None
            z_sol = np.array([0.0, 0.0])
            z_opt = self.runOptimization(max_iter, step_size, box_bounds, use_sa=use_sa)
            print("True optimum: ", z_sol)
            print("From optimizer: ", z_opt)
            self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

            # Box constrained around -1, 1
            box_bounds = [np.array([-1, -1]), np.array([1, 1])]
            z_sol = np.array([0.0, 0.0])
            z_opt = self.runOptimization(max_iter, step_size, box_bounds, use_sa=use_sa)
            print("True optimum: ", z_sol)
            print("From optimizer: ", z_opt)
            self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

            # Box constrained [2,3] optimal should be 2
            box_bounds = [np.array([2, 2]), np.array([3, 3])]
            z_sol = np.array([2.0, 2.0])
            z_opt = self.runOptimization(max_iter, step_size, box_bounds, use_sa=use_sa)
            print("True optimum: ", z_sol)
            print("From optimizer: ", z_opt)
            self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

            # Only one constrained variable
            box_bounds = [np.array([2, -np.inf]), np.array([3, np.inf])]
            z_sol = np.array([2.0, 0.0])
            z_opt = self.runOptimization(max_iter, step_size, box_bounds, use_sa=use_sa)
            print("True optimum: ", z_sol)
            print("From optimizer: ", z_opt)
            self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

            z_constraint = dl.Function(self.Vz).vector()
            a = 1.3
            c = np.array([2.1, 0.4])
            z_constraint.set_local(np.array([2.1, 0.4]))
            z_sol = a * c /np.inner(c,c)
            constraint = InnerProductEqualityConstraint(z_constraint, a)
            z_opt = self.runOptimization(max_iter, step_size, None, other_constraint=constraint, use_sa=use_sa)
            print("True optimum: ", z_sol)
            print("From optimizer: ", z_opt)
            self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)


if __name__ == "__main__":
    unittest.main()

