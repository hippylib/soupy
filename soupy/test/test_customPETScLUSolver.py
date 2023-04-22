import unittest 
import dolfin as dl
import sys 
sys.path.append("../solver")
from customPETScLUSolver import CustomPETScLUSolver

class TestCustomPETScLUSolver(unittest.TestCase):
    def setUp(self):
        self.mesh = dl.UnitSquareMesh(20,20)
        self.method = "mumps"
        self.tol = 1e-12

    def testSolver(self):
        V = dl.FunctionSpace(self.mesh, "CG", 2)
        u_trial = dl.TrialFunction(V)
        u_test = dl.TestFunction(V)
        u_fun = dl.Function(V)


        a_form = dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx + u_trial*u_test*dl.dx
        b_form = dl.Constant(1.0) * u_test * dl.dx 

        bcs = dl.DirichletBC(V, dl.Constant(0.0), "on_boundary")

        A = dl.PETScMatrix()
        b = dl.PETScVector()
        dl.assemble_system(a_form, b_form, bcs=bcs, A_tensor=A, b_tensor=b)
        
        u = u_fun.vector()
        u_custom = u.copy()

        diff = dl.Function(V).vector()

        A_solver_custom = CustomPETScLUSolver(method=self.method)
        A_solver_custom.set_operator(A)
        A_solver_custom.solve(u_custom, b)

        A_solver = dl.PETScLUSolver(A)
        A_solver.solve(u,b)
        
        diff.axpy(1.0, u)
        diff.axpy(-1.0, u_custom)
        print("Error in computed solution against fenics solver: %.3e" %(diff.norm('l2')))
        self.assertTrue(diff.norm('l2') < self.tol)


if __name__ == "__main__":
    unittest.main()
