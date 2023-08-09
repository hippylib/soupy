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
import os
import sys
import math

import numpy as np
import dolfin as dl
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)

sys.path.append('../../')
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, \
    InexactNewtonCG_ParameterList, InexactNewtonCG

class QuadraticCost:
    def __init__(self, Vz, dim):
        self.dim = dim
        self.Vz = Vz

    def generate_vector(self, component=CONTROL):
        assert component == CONTROL
        return dl.Function(self.Vz).vector()

    def cost(self, z, order=0, sample_size=1, rng=None):
        self.z = z 
        return z.inner(z)

    def grad(self, g):
        g.zero()
        g.axpy(2.0, self.z)
        return g.inner(g)

    def hessian(self, zhat, Hzhat):
        Hzhat.zero()
        Hzhat.axpy(1.0, zhat)




class TestInexactNewtonCG(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(10, 10)

    def runOptimization(self, cost, max_iter, globalization_type):
        newton_param = InexactNewtonCG_ParameterList()
        newton_param["max_iter"] = max_iter
        newton_param["globalization"] = globalization_type
        newton_solver = InexactNewtonCG(cost, newton_param)

        z0 = dl.Function(cost.Vz).vector()
        hp.parRandom.normal(1.0, z0)
        z_opt, result = newton_solver.solve(z0)
        print(newton_solver.termination_reasons[newton_solver.reason])
        return z_opt.get_local()

    def testQuadraticCost(self):
        """
        Test that optimization is carried out correctly 
        """

        # Unconstrained quadratic case
        tol = 1e-3
        dim = 10
        max_iter = 100
        z_sol = np.zeros(dim)

        Vz = dl.VectorFunctionSpace(self.mesh, "R", 0, dim=dim)
        cost = QuadraticCost(Vz, dim)

        print(80*"-")
        print("Unconstrained quadratic with line search")
        z_opt = self.runOptimization(cost, max_iter, "LS")
        print("Optimal z: ", z_opt)
        print("True z: ", z_sol)
        print(80*"-")
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

        print(80*"-")
        print("Unconstrained quadratic with trust region")
        z_opt = self.runOptimization(cost, max_iter, "TR")
        print("Optimal z: ", z_opt)
        print("True z: ", z_sol)
        print(80*"-")
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)


if __name__ == "__main__":
    unittest.main()

