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
import sys 
import os 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 

sys.path.append('../../')
from soupy import CGSolverSteihaug, CGSolverSteihaug_ParameterList


def make_linear_system(V, A_tensor, M_tensor, b_tensor, eps):
    """
    Assembles the linear system for a symmetric elliptic system \
    """

    u_trial = dl.TrialFunction(V)
    u_test = dl.TestFunction(V)

    a_form = dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx \
            + dl.Constant(eps)*u_trial*u_test*dl.dx
    b_form = dl.Constant(1.0) * u_test * dl.dx 

    m_form = dl.inner(dl.grad(u_trial), dl.grad(u_test)) * dl.dx 

    boundary_term = dl.Expression("x[0]", degree=1)
    bcs = dl.DirichletBC(V, boundary_term, "on_boundary")

    dl.assemble_system(m_form, b_form, bcs=bcs, A_tensor=M_tensor, b_tensor=b_tensor)
    dl.assemble_system(a_form, b_form, bcs=bcs, A_tensor=A_tensor, b_tensor=b_tensor)





class TestCGSolverSteihaug(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(20,20)
        self.V = dl.FunctionSpace(self.mesh, "CG", 2)
        self.tol = 1e-6


    def compareSolves(self, A, b, P=None, backend="dolfin"):
        cg_solver = CGSolverSteihaug()
        cg_solver.set_operator(A)
        cg_solver.set_preconditioner(P)

        dl_solver = dl.PETScLUSolver(A)

        u_true = dl.Function(self.V).vector()
        u_cg = dl.Function(self.V).vector()
        diff = dl.Function(self.V).vector()

        cg_solver.solve(u_cg, b)
        dl_solver.solve(u_true, b)

        diff.axpy(1.0, u_cg)
        diff.axpy(-1.0, u_true)
        print("solve with %s backend: Error in computed solution against fenics solver: %.3e" %(backend, diff.norm('l2')))
        self.assertTrue(diff.norm('l2') < self.tol)
        return cg_solver.iter

    def testSolver(self):
        """
        Test solution for linear systems with and without preconditioners 
        """

        eps = 1.0 
        A = dl.PETScMatrix()
        M = dl.PETScMatrix()
        b = dl.PETScVector()

        make_linear_system(self.V, A, M, b, eps)

        A_solver = dl.PETScLUSolver(A)
        M_solver = dl.PETScLUSolver(M)

        print("Compare solve without preconditioner")
        iter = self.compareSolves(A, b)

        print("Compare solve with preconditioner")
        iter = self.compareSolves(A, b, P=M_solver)

        print("Compare solve with exact preconditioner")
        iter = self.compareSolves(A, b, P=A_solver)

        print("Assert that convegence occurs with one iteration")
        self.assertEqual(iter, 1)

if __name__ == "__main__":
    unittest.main()
    
