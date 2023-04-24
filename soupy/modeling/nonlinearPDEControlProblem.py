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

import sys, os
import numpy as np
import dolfin as dl

import hippylib as hp 

from .variables import CONTROL
from .PDEControlProblem import PDEVariationalControlProblem
from .nonlinearStateProblem import NonlinearStateProblem

class NonlinearPDEControlProblem(PDEVariationalControlProblem):
    def __init__(self, Vh, nonlinear_residual, bc, bc0, state_solver, lu_method="default"):
        # assert for class assumptions here
        assert id(Vh[hp.STATE]) == id(Vh[hp.ADJOINT]), print('Need to have same STATE and ADJOINT spaces')
        assert len(Vh) == 4
        assert Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')

        self.nonlinear_residual = nonlinear_residual 
        self.state_solver = state_solver
        self.nonlinear_problem = NonlinearStateProblem(Vh[hp.STATE], nonlinear_residual, bc, bc0)
        self.lu_method = lu_method

        self.Vh = Vh
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
        self.A  = None
        self.At = None
        self.C = None
        self.Cz = None
        self.Wmu = None
        self.Wmm = None
        self.Wzu = None
        self.Wzz = None
        self.Wuu = None
        
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}
        self.n_linear_solves = 0 

    def varf_handler(self, u, m, p, z):
        return self.nonlinear_residual.residual(u, p, m, z)

    def solveFwd(self, u, x):
        m_fun = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        num_iter, converged = self.state_solver.solve(self.nonlinear_problem, u, x[hp.STATE], extra_args=[m_fun, z_fun]) 
        self.n_linear_solves += num_iter

    
