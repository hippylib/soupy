import math
import numpy as np
import dolfin as dl
import sys
import os
from mpi4py import MPI

sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

# sys.path.append(os.environ.get('HIPPYCONTROL_PATH'))
# import hippycontrol as hc
from ..modeling.variables import CONTROL

def SteepestDescent_ParameterList():
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-8, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-12, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [500, "maximum number of iterations"]
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    parameters["print_level"]           = [0, "Print info on screen"]
    parameters["alpha"]                 = [1., "Initial scaling alpha"]
    return hp.ParameterList(parameters)

class SteepestDescent:

    """
    Gradient Descent to solve optimization under uncertainty problems
    Globalization is performed using the Armijo sufficient reduction condition (backtracking).
    The stopping criterion is based on a control on the norm of the gradient.

    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient.

    More specifically the model object should implement following methods:
       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint.
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately.
       - :code:`solveFwd(out, x)` -> solve the possibly non linear forward problem.
       - :code:`solveAdj(out, x)` -> solve the linear adjoint problem.
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm.
       - :code:`Rsolver()`          --> A solver for the regularization term.

    Type :code:`help(Model)` for additional information.
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of the step less than tolerance",     #3
                           ]

    def __init__(self, cost_functional, parameters = SteepestDescent_ParameterList()):
        """
        Initialize the Stochastic Gradient Descent solver. Type :code:`SGD_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        """
        self.cost_functional = cost_functional
        self.parameters = parameters

        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        self.random_seed = 1

    def solve(self, z, box_bounds=None, constraint_projection=None):
        """
        Solve the constrained optimization problem with initial guess :code:`x = [u,a,p]`. Return the solution :code:`[u,a,p]`.
        .. note:: :code:`x` will be overwritten.

        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        c_armijo = self.parameters["c_armijo"]
        max_backtracking_iter = self.parameters["max_backtracking_iter"]
        alpha = self.parameters["alpha"]
        print_level = self.parameters["print_level"]

        mpi_world_rank = MPI.COMM_WORLD.Get_rank()

        self.it = 0
        self.converged = False
        self.ncalls += 1


        if box_bounds is not None:
            if hasattr(box_bounds[0], "get_local"):
                param_min = box_bounds[0].get_local()    #Assume it is a dolfin vector
            else:
                param_min = box_bounds[0]*np.ones_like(z.get_local()) #Assume it is a scalar
            if hasattr(box_bounds[1], "get_local"):
                param_max = box_bounds[1].get_local()    #Assume it is a dolfin vector
            else:
                param_max = box_bounds[1]*np.ones_like(z.get_local()) #Assume it is a scalar

        # Initialize vectors
        zhat = self.cost_functional.generate_vector(CONTROL)
        g = self.cost_functional.generate_vector(CONTROL)
        dz = self.cost_functional.generate_vector(CONTROL)
        z_star = self.cost_functional.generate_vector(CONTROL)

        # Initialize costs
        costs = [] 
        cost_old = self.cost_functional.cost(z, order=1)
        costs.append(cost_old)

        if print_level >= 0 and mpi_world_rank == 0:
            print( "\n{0:3} {1:15} {2:15} {3:15} {4:15}".format(
                  "It", "cost", "||g||L2", "||dz||L2", "alpha") )
            print( "{0:3d} {1:15e}".format(
                    self.it, cost_old))

        gradnorms = [] 
        n_backtracks = [] 
        
        # Run optimization 
        while (self.it < max_iter) and (self.converged == False):
            gradnorm = self.cost_functional.costGrad(z, g)
            if gradnorm is None:
                gradnorm = np.sqrt(g.inner(g))
            
            gradnorms.append(gradnorm)

            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)

            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            zhat.zero()
            zhat.axpy(-1.0, g)
            
            # initialize variables for backtracking 
            scale_alpha = 1.0
            descent = False
            n_backtrack = 0

            g_zhat = g.inner(zhat)
            # Using SAA and backtracking 
            if print_level >= 0 and mpi_world_rank == 0:
                print("With backtracking")
            if print_level >= 1 and mpi_world_rank == 0:
                print("\tCurrent cost: %g" %cost_old)
            
            # Begin backtracking 
            while not descent and (n_backtrack < max_backtracking_iter):
                # Update the optimization variable 
                z_star.zero()
                z_star.axpy(1.0, z)
                z_star.axpy(scale_alpha * alpha, zhat)
                if box_bounds is not None:
                    z_star.set_local(np.maximum(z_star.get_local(), param_min))
                    z_star.set_local(np.minimum(z_star.get_local(), param_max))
                if constraint_projection is not None:
                    constraint_projection.project(z_star)

                # compute the cost 
                cost_new = self.cost_functional.cost(z_star, order=0)
                if print_level >= 1:
                    print("\tBacktracking cost for step size %g: \t%g" %(scale_alpha, cost_new))
                                
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + scale_alpha * alpha * c_armijo * g_zhat):
                    descent = True
                else:
                    n_backtrack += 1
                    scale_alpha *= 0.5
            
            n_backtracks.append(n_backtrack)

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break

            # Compute size of step
            dz.zero()
            dz.axpy(1.0, z_star)
            dz.axpy(-1.0, z)
            dz_norm = np.sqrt(dz.inner(dz))
            gdz = g.inner(dz)

            
            # Update z and compute cost 
            z.zero()
            z.axpy(1.0, z_star)
            cost_old = self.cost_functional.cost(z, order=1)
            costs.append(cost_old)

            # if(print_level >= 0) and (self.it == 1):
            #     print( "\n{0:3} {1:15} {2:15} {3:15} {4:15}".format(
            #           "It", "cost", "||g||L2", "||dz||L2", "alpha") )

            if print_level >= 0 and mpi_world_rank == 0:
                print( "{0:3d} {1:15e} {2:15e} {3:15e} {4:15}".format(
                        self.it, cost_old, gradnorm, dz_norm, scale_alpha * alpha) )

            if -gdz < self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old

        result = dict()
        result["costs"] = costs
        result["gradnorms"] = gradnorms
        result["n_backtracks"] = n_backtracks
        result["termination_reason"] = SteepestDescent.termination_reasons[self.reason]
        
        if print_level >= 0 and mpi_world_rank == 0:
            print(SteepestDescent.termination_reasons[self.reason])


        return z, result
