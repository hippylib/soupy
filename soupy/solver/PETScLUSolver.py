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


import dolfin as dl 
from petsc4py import PETSc
from mpi4py import MPI


class PETScLUSolver:
    """
    LU solver for linear systems :math:`Ax = b` and :math:`A^Tx = b`. \
        It is a wrapper for :py:class:`dolfin.PETScLUSolver` \
        providing custom implementations for methods :code:`solve_transpose` \
        and :code:`set_operator` if they are unavailable in the \
        :code:`dolfin` version being used.
    """
    def __init__(self, mpi_comm=MPI.COMM_WORLD, method="mumps"):
        """
        Constructor:

        :param mpi_comm: MPI Communicator for the linear system
        :type mpi_comm: :py:class:`MPI.Comm`
        :param method: LU method 
        :type method: str
        """
        self.mpi_comm = mpi_comm 
        self.method = method 
        self.solver = dl.PETScLUSolver(mpi_comm, method)
        self.ksp = self.solver.ksp()

    def set_operator(self, A):
        """
        Set the linear operator 

        :param A: Matrix for the solves
        :type A: :py:class:`dolfin.Matrix` or :py:class:`dolfin.PETScMatrix`
        """

        if hasattr(A, 'mat'):
            self.ksp.setOperators(A.mat())
        else:
            self.ksp.setOperators(dl.as_backend_type(A).mat())

    def solve(self, x, b):
        """
        Solve the linear system :math:`Ax = b` \
            and stores result to :code:`x`

        :param x: Solution vector
        :type x: :py:class:`dolfin.Vector` or :py:class:`dolfin.PETScVector`
        :param b: Right hand side vector
        :type b: :py:class:`dolfin.Vector` or :py:class:`dolfin.PETScVector`
        """

        self.solver.solve(x, b)

    def solve_transpose(self, x, b):
        """
        Solve the linear system :math:`A^T x = b` \
            and stores result to :code:`x`

        :param x: Solution vector
        :type x: :py:class:`dolfin.Vector` or :py:class:`dolfin.PETScVector`
        :param b: Right hand side vector
        :type b: :py:class:`dolfin.Vector` or :py:class:`dolfin.PETScVector`
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
