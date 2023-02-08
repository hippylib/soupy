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
from hippycontrol import STATE, PARAMETER, ADJOINT, CONTROL, SteepestDescent, SteepestDescent_ParameterList, InnerProductEqualityConstraint

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


class TestSteepesetDescent(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(10, 10)
        self.dim = 2
        self.Vz = dl.VectorFunctionSpace(self.mesh, "R", 0, dim=self.dim)
        self.cost = QuadraticCost(self.Vz, self.dim)
        self.max_backtracking_iter = 20 

    
    def testResultOutput(self):
        steepestDescent_param = SteepestDescent_ParameterList()
        steepestDescent_param["max_iter"] = 10
        steepestDescent_param["alpha"] = 0.95
        steepestDescent_param["max_backtracking_iter"] = self.max_backtracking_iter
        steepestDescent_solver = SteepestDescent(self.cost, steepestDescent_param)

        z0 = dl.Function(self.Vz).vector()
        z0.set_local(np.random.randn(self.dim)*10)
        z_opt, results = steepestDescent_solver.solve(z0)
        
        self.assertTrue("costs" in results.keys())
        self.assertTrue("gradnorms" in results.keys())
        self.assertTrue("n_backtracks" in results.keys())
        self.assertTrue("termination_reason" in results.keys())
    


    def runOptimization(self, max_iter, step_size, box_bounds, other_constraint=None):
        steepestDescent_param = SteepestDescent_ParameterList()
        steepestDescent_param["max_iter"] = max_iter
        steepestDescent_param["alpha"] = step_size
        steepestDescent_param["max_backtracking_iter"] = self.max_backtracking_iter

        steepestDescent_solver = SteepestDescent(self.cost, steepestDescent_param)

        z0 = dl.Function(self.Vz).vector()
        z0.set_local(np.random.randn(self.dim)*10)
        if box_bounds is not None:
            param_min = box_bounds[0]
            param_max = box_bounds[1]
            z0.set_local(np.maximum(z0.get_local(), param_min))
            z0.set_local(np.minimum(z0.get_local(), param_max))

        if other_constraint is not None:
            other_constraint.project(z0)


        z_opt, _ = steepestDescent_solver.solve(z0, box_bounds, other_constraint)
        print("Optimal z: ")
        print(z_opt.get_local())
        return z_opt.get_local()


    def testOptimization(self):
        # Unconstrained quadratic case
        tol = 1e-3
        max_iter = 100
        step_size = 0.5
        box_bounds = None
        z_sol = np.array([0.0, 0.0])
        z_opt = self.runOptimization(max_iter, step_size, box_bounds)
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

        # Box constrained around -1, 1
        box_bounds = [np.array([-1, -1]), np.array([1, 1])]
        z_sol = np.array([0.0, 0.0])
        z_opt = self.runOptimization(max_iter, step_size, box_bounds)
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

        # Box constrained [2,3] optimal should be 2
        box_bounds = [np.array([2, 2]), np.array([3, 3])]
        z_sol = np.array([2.0, 2.0])
        z_opt = self.runOptimization(max_iter, step_size, box_bounds)
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

        # Only one constrained variable
        box_bounds = [np.array([2, -np.inf]), np.array([3, np.inf])]
        z_sol = np.array([2.0, 0.0])
        z_opt = self.runOptimization(max_iter, step_size, box_bounds)
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)

        z_constraint = dl.Function(self.Vz).vector()
        a = 1.3
        c = np.array([2.1, 0.4])
        z_constraint.set_local(np.array([2.1, 0.4]))
        z_sol = a * c /np.inner(c,c)
        constraint = InnerProductEqualityConstraint(z_constraint, a)
        z_opt = self.runOptimization(max_iter, step_size, None, other_constraint=constraint)
        self.assertTrue(np.linalg.norm(z_sol-z_opt) < tol)


if __name__ == "__main__":
    unittest.main()
