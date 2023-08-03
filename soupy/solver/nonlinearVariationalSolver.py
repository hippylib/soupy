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

class NonlinearVariationalSolver:
    """
    Solver for the nonlinear variational problem 

        .. math:: \\text{Find } u \in U \qquad r(u,v) = 0 \quad \\forall v \in V

    The user must provide the variational forms for the residual form, \
        an initial guess, boundary conditions, and optionally, \
        variational forms for the Jacobian and backtracking cost functional.

    This wraps the `dolfin.NonlinearVariationalProblem` and `dolfin.NonlinearVariationalSolver` \
        classes using a different function signature 
    """
    def __init__(self, parameters=None):
        """
        Constructor:

        :param parameters: Solver parameters for the `dolfin.NonlinearVariationalSolver` class 
        :type parameters: dict 

        Class will update the default `dolfin.NonlinearVariationalSolver` parameters with the \
            data in :code:`parameters` if it is provided. Otherwise, default parameters \
            will be used
        """
        self.parameters = parameters

    def solve(self, residual_form, u, bcs, J_form, form_compiler_parameters=None):
        """
        Solve the nonlinear variational system 

        .. math:: \\text{Find } u \in U \qquad r(u,v) = 0 \quad \\forall v \in V

        given using the `dolfin.NonlinearVariationalSolver` with a supplied initial guess

        :param residual_form: Variational form for the residual 
        :param u: Initial guess for the solution and function used in :code:`residual_form`
        :type u: :py:class:`dolfin.Function`
        :param bcs: List of boundary conditions 
        :type bcs: list 
        :param J_form: Variational form for the Jacobian of the residual 
        :param energy_form: Optional variational form for the energy functional. 
            If supplied, uses this as the backtracking cost
        :param form_compiler_parameters: Optional form compiler_parameters
        """
        nonlinear_problem = dl.NonlinearVariationalProblem(residual_form, u, bcs, J_form, 
                                                 form_compiler_parameters=form_compiler_parameters)
        solver = dl.NonlinearVariationalSolver(nonlinear_problem)
        if self.parameters is not None:
            solver.parameters.update(self.parameters)

        num_iters, converged = solver.solve()
        return num_iters, converged

