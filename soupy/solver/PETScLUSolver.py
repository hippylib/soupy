import dolfin as dl 
from petsc4py import PETSc
from mpi4py import MPI


class PETScLUSolver:
    """
    Wrapper for `dolfin.PETScLUSolver` with methods
    :code:`solve_transpose` and :code:`set_operator`
    """
    def __init__(self, mpi_comm=MPI.COMM_WORLD, method="mumps"):
        self.mpi_comm = mpi_comm 
        self.method = method 
        self.solver = dl.PETScLUSolver(mpi_comm, method)
        self.ksp = self.solver.ksp()

    def set_operator(self, A):
        if hasattr(A, 'mat'):
            self.ksp.setOperators(A.mat())
        else:
            self.ksp.setOperators(dl.as_backend_type(A).mat())

    def solve(self, x, b):
        """
        Solve the linear system 

        .. math:: `Ax = b`

        and stores result to :code:`x`
        """

        self.solver.solve(x, b)

    def solve_transpose(self, x, b):
        """
        Solve the linear system 

        .. math:: `A^T x = b`

        and stores result to :code:`x`
        """
        if hasattr(self.solver, 'solve_transpose'):
            self.solver.solveTranspose(x, b)
        else:
            self._custom_solveTranspose(x, b)


    def _custom_solveTranspose(self, x, b):
        if hasattr(b, 'vec'):
            b_vec = b.vec()
        else:
            b_vec = dl.as_backend_type(b).vec()

        if hasattr(x, 'vec'):
            # x already has vec. Can directly solve to x.vec()
            self.ksp.solveTranspose(b_vec, x.vec())

        else:
            # x has no vec object. Need to make one, and then 
            # store the result to x
            x_petsc = dl.as_backend_type(x)
            self.ksp.solveTranspose(b_vec, x_petsc.vec())
            
            # Now store to the input x
            x.set_local(x_petsc.get_local())
            x.apply("")
