import unittest

import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl

import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

sys.path.append('../../')
from soupy import STATE, PARAMETER, ADJOINT, CONTROL, \
        NewtonBacktrackSolver, NonlinearVariationalSolver

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def residual(u, v):
    return dl.inner(dl.grad(u), dl.grad(v)) * dl.dx + u**3 * v * dl.dx  - dl.Constant(1.0) * v * dl.dx 

class TestPoissonPDEControlProblem(unittest.TestCase):
    def setUp(self):
        self.reltol = 1e-3
        self.nx = 20
        self.ny = 20
        mesh = dl.UnitSquareMesh(self.nx, self.ny)
        self.Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)

        u_on_boundary = dl.Expression("x[0]", degree=1)
        self.bc = dl.DirichletBC(self.Vh_STATE, u_on_boundary, u_boundary)

    def _solve_by_dolfin_solve(self):
        u_fun = dl.Function(self.Vh_STATE)
        v = dl.TestFunction(self.Vh_STATE)
        res_form = residual(u_fun, v)
        J_form = dl.derivative(res_form, u_fun)
        dl.solve(res_form == 0, u_fun, bcs=self.bc, J=J_form)
        return u_fun

    def _solve_by_nonlinear_solver(self):
        u_fun = dl.Function(self.Vh_STATE)
        v = dl.TestFunction(self.Vh_STATE)
        res_form = residual(u_fun, v)
        J_form = dl.derivative(res_form, u_fun)
        solver = NonlinearVariationalSolver()
        num_iter, converged = solver.solve(res_form, u_fun, self.bc, J_form)
        return u_fun, num_iter, converged

    def _solve_by_nonlinear_snes_solver(self):
        params = {'nonlinear_solver': 'snes',
            'snes_solver':
            {
                'linear_solver'           : 'mumps',
                'absolute_tolerance'      : 1e-10,
                'relative_tolerance'      : 1e-10,
                'maximum_iterations'      : 20,
             }
         }
        u_fun = dl.Function(self.Vh_STATE)
        v = dl.TestFunction(self.Vh_STATE)
        res_form = residual(u_fun, v)
        J_form = dl.derivative(res_form, u_fun)
        solver = NonlinearVariationalSolver(params)
        num_iter, converged = solver.solve(res_form, u_fun, self.bc, J_form)
        return u_fun, num_iter, converged

    def _solve_by_newton_solver(self):
        u_fun = dl.Function(self.Vh_STATE)
        v = dl.TestFunction(self.Vh_STATE)
        res_form = residual(u_fun, v)
        J_form = dl.derivative(res_form, u_fun)
        solver = NewtonBacktrackSolver()
        solver.parameters['print_level'] = 0 
        num_iter, converged = solver.solve(res_form, u_fun, self.bc, J_form)
        return u_fun, num_iter, converged

    def testNonlinearSolver(self):
        """
        Test the wrapper to the nonlinaer solver
        """
        u_dl = self._solve_by_dolfin_solve()
        u_nl, n_iter, converged = self._solve_by_nonlinear_solver()
        diff_nl = u_dl.vector() - u_nl.vector()
        rel_err_nl = np.sqrt(diff_nl.inner(diff_nl)/u_dl.vector().inner(u_dl.vector()))
        print("Relative solver errors, dolfin vs nonlinear: %g" %(rel_err_nl))
        print("N_iters: %d" %(n_iter), "converged: ",  converged)
        self.assertTrue(rel_err_nl < self.reltol)
        self.assertTrue(n_iter > 0)
        self.assertTrue(converged)

    def testSNESSolver(self):
        """
        Test the wrapper to the nonlinear solver with parameter updates \
            To use the SNES solver instead
        """
        u_dl = self._solve_by_dolfin_solve()
        u_nl, n_iter, converged = self._solve_by_nonlinear_snes_solver()
        diff_nl = u_dl.vector() - u_nl.vector()
        rel_err_nl = np.sqrt(diff_nl.inner(diff_nl)/u_dl.vector().inner(u_dl.vector()))
        print("Relative solver errors, dolfin vs SNES: %g" %(rel_err_nl))
        print("N_iters: %d" %(n_iter), "converged: ",  converged)
        self.assertTrue(rel_err_nl < self.reltol)
        self.assertTrue(n_iter > 0)
        self.assertTrue(converged)

    def testNewtonSolver(self):
        """
        Test the custom Newton nonlinear solver 
        """
        u_dl = self._solve_by_dolfin_solve()
        u_nt, n_iter, converged = self._solve_by_newton_solver()
        diff_nt = u_dl.vector() - u_nt.vector()
        rel_err_nt = np.sqrt(diff_nt.inner(diff_nt)/u_dl.vector().inner(u_dl.vector()))
        print("Relative solver errors, dolfin vs custom backtrack newton: %g" %(rel_err_nt))
        print("N_iters: %d" %(n_iter), "converged: ",  converged)
        self.assertTrue(rel_err_nt < self.reltol)
        self.assertTrue(n_iter > 0)
        self.assertTrue(converged)

if __name__ == "__main__":
    unittest.main()

