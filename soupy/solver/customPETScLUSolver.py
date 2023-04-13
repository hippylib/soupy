import dolfin as dl 
from petsc4py import PETSc
from mpi4py import MPI


class CustomPETScLuSolver:
    def __init__(self, mpi_comm=MPI.COMM_WORLD, method="mumps"):
        self.mpi_comm = mpi_comm 
        self.ksp = PETSc.KSP().create(comm=self.mpi_comm)
        self.ksp.setType('preonly')
        self.pc = self.ksp.getPC()
        self.pc.setType('lu') 
        opts = PETSc.Options()
        opts['pc_factor_mat_solver_type'] = method
        self.pc.setFromOptions()

    def set_operator(self, A):
        A_mat = A.mat()
        self.ksp.setOperators(A_mat, A_mat) 

    def solve(self, x, b):
        print("Calling custom solve")
        b_vec = b.vec()
        x_vec = x.vec()
        self.ksp.solve(b_vec, x_vec) 

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    mesh = dl.UnitSquareMesh(20,20)
    V = dl.FunctionSpace(mesh, "CG", 2)
    u_trial = dl.TrialFunction(V)
    u_test = dl.TestFunction(V)
    u_fun = dl.Function(V)
    u = u_fun.vector()

    a_form = dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx + u_trial*u_test*dl.dx
    b_form = dl.Constant(1.0) * u_test * dl.dx 

    bcs = dl.DirichletBC(V, dl.Constant(0.0), "on_boundary")

    A = dl.PETScMatrix()
    b = dl.PETScVector()
    dl.assemble_system(a_form, b_form, bcs=bcs, A_tensor=A, b_tensor=b)
    
    A_solver = dl.PETScLUSolver(A)
    A_solver.solve(u,b)

    plt.figure()
    p = dl.plot(u_fun)
    plt.colorbar(p)
    plt.title("Fenics built in")

    A_solver_custom = CustomPETScLuSolver()
    A_solver_custom.set_operator(A)
    A_solver_custom.solve(u, b)

    plt.figure()
    p = dl.plot(u_fun)
    plt.colorbar(p)
    plt.title("Custom")
    plt.show()
