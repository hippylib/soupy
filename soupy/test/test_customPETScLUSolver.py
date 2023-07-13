import unittest 
import dolfin as dl
import sys 
import os 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 

sys.path.append('../../')
from soupy import PETScLUSolver


def make_linear_system(V, A_tensor, b_tensor):
    """
    Assembles the linear system for a linear ADR problem. \
        This is not symmetric 
    """

    u_trial = dl.TrialFunction(V)
    u_test = dl.TestFunction(V)

    a_form = dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx \
            + dl.inner(dl.Constant((1.0, 1.0)), dl.grad(u_trial))*u_test*dl.dx \
            + u_trial*u_test*dl.dx
    b_form = dl.Constant(1.0) * u_test * dl.dx 

    boundary_term = dl.Expression("x[0]", degree=1)
    bcs = dl.DirichletBC(V, boundary_term, "on_boundary")

    dl.assemble_system(a_form, b_form, bcs=bcs, A_tensor=A_tensor, b_tensor=b_tensor)


class TestCustomPETScLUSolver(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(20,20)
        self.method = "mumps"
        self.V = dl.FunctionSpace(self.mesh, "CG", 2)
        self.tol = 1e-12


    def compareSolve(self, backend="dolfin"):
        """
        Compares the solution of using custom solver against \
            Standard solver
        """
        A = dl.PETScMatrix()
        b = dl.PETScVector()

        make_linear_system(self.V, A, b)
        
        u_fun = dl.Function(self.V)
        u = u_fun.vector()
        u_custom = u.copy()

        diff = dl.Function(self.V).vector()

        A_solver_custom = PETScLUSolver(method=self.method)
        A_solver_custom.set_operator(A)
        A_solver_custom.solve(u_custom, b)

        A_solver = dl.PETScLUSolver(A)
        A_solver.solve(u,b)
        
        diff.axpy(1.0, u)
        diff.axpy(-1.0, u_custom)
        print("solve with %s: Error in computed solution against fenics solver: %.3e" %(backend, diff.norm('l2')))
        self.assertTrue(diff.norm('l2') < self.tol)
    
    def compareSolveTranspose(self, backend="dolfin"):
        """
        Compares the solution of using custom solver against \
            Standard solver for solving the transposed system
        """
        if backend == "dolfin":
            A = dl.Matrix()
            b = dl.Vector()
        elif backend == "petsc":
            A = dl.PETScMatrix()
            b = dl.PETScVector()
        else:
            raise ValueError("Not a valid backend")

        make_linear_system(self.V, A, b)
        At = dl.as_backend_type(hp.Transpose(A))
        u_fun = dl.Function(self.V)
        u = u_fun.vector()
        u_custom = u.copy()

        diff = dl.Function(self.V).vector()
        
        # Use the solveTranspose with custom solver 
        A_solver_custom = PETScLUSolver(method=self.method)
        A_solver_custom.set_operator(A)
        A_solver_custom.solve_transpose(u_custom, b)

        # a standard transpose solver 
        At_solver = dl.PETScLUSolver(At)
        At_solver.solve(u,b)
        
        diff.axpy(1.0, u)
        diff.axpy(-1.0, u_custom)
        print("solve_tranpose with %s: Error in computed solution against fenics solver: %.3e" %(backend, diff.norm('l2')))
        self.assertTrue(diff.norm('l2') < self.tol)

    def testSolveDolfin(self):
        self.compareSolve(backend="dolfin")

    def testSolveTransposeDolfin(self):
        self.compareSolveTranspose(backend="dolfin")

    def testSolvePETSc(self):
        self.compareSolve(backend="petsc")

    def testSolveTransposePETSc(self):
        self.compareSolveTranspose(backend="petsc")


if __name__ == "__main__": 
    unittest.main()


