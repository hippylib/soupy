import dolfin as dl 

class NonlinearVariationalSolver:
    """
    Solver for the nonlinear variational problem 

        .. math:: r(u,v) = 0 \\forall v

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

        .. math:: r(u,v) = 0 \\forall v

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

