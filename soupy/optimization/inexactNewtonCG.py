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


import math

from hippylib import ParameterList 

from ..modeling.controlCostHessian import ControlCostHessian
from ..modeling.variables import CONTROL
from .cgSolverSteihaug import CGSolverSteihaug


def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]
    
    return ParameterList(parameters)

def InexactNewtonCG_ParameterList():
    """
    Generate a ParameterList for InexactNewtonCG.
    type: :code:`InexactNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdz_tolerance"]         = [1e-18, "we converge when (g,dz) <= gdz_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["cg_max_iter"]           = [100, "Maximum CG iterations"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]
    
    return ParameterList(parameters)
  
    

class InexactNewtonCG:
    
    """
    Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:

    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
        - :code:`cost(z)` -> evaluate the cost functional
        - :code:`costGrad(g) -> evaluate the gradient and store to :code:`g`
        - :code:`costHessian(zhat, Hzhat) -> apply the cost Hessian
       
    Type :code:`help(Model)` for additional information
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, dz) less than tolerance"       #3
                           ]
    
    def __init__(self, cost_functional, parameters=InexactNewtonCG_ParameterList(), 
                preconditioner=None,
                norm_weighting=None,
                callback = None):
        """
        Constructor for the SGD solver

        :param cost_functional: The cost functional object
        :type cost_functional: :py:class:`soupy.ControlCostFunctional` or similar
        :param parameters: The parameters of the solver.
            Type :code:`InexactNewtonCG_ParameterList().showMe()` for list of default parameters
            and their descriptions.
        :type parameters: :py:class:`hippylib.ParameterList`.
        :param preconditioner: Optional preconditioner for the Hessian inverse with method :code:`mult`
            and :code:`solve` Has optional method :code:`setLinearizationPoint`
        :param norm_weighting: Weighting matrix for the norm with method :code:`mult` 
        """

        self.cost_functional = cost_functional
        self.parameters = parameters
        self.preconditioner = preconditioner
        self.norm_weighting = norm_weighting 

        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        self.callback = callback

        
    def solve(self, z):
        """
        Solve the constrained optimization problem with initial guess :code:`z`

        :param z: The initial guess
        :type z: :py:class:`dolfin.Vector` 

        :return: The optimization solution :code:`z` and a dictionary of information

        .. note:: The input :code:`z` will be overwritten 
        """
        if self.cost_functional is None:
            raise TypeError("Cost functional cannot be of type None")
        if self.parameters["globalization"] == "LS":
            return self._solve_ls(z)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(z)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_ls(self, z):
        """
        Solve the constrained optimization problem with initial guess :code:`x`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        cg_max_iter         = self.parameters["cg_max_iter"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]
        
        # self.model.solveFwd(x[STATE], x)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        zhat = self.cost_functional.generate_vector(CONTROL)    
        zstar = self.cost_functional.generate_vector(CONTROL)    
        mg = self.cost_functional.generate_vector(CONTROL)
                
        cost_old = self.cost_functional.cost(z, order=2)
        
        while (self.it < max_iter) and (self.converged == False):
            # self.model.solveAdj(x[ADJOINT], x)
            # self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            # gradnorm = self.model.evalGradientParameter(x, mg)

            # Compute the gradient 
            gradnorm = self.cost_functional.costGrad(mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            cost_hessian = ControlCostHessian(self.cost_functional)
            solver = CGSolverSteihaug()
            solver.set_operator(cost_hessian)

            if self.preconditioner is not None:
                if hasattr(self.preconditioner, 'setLinearizationPoint'):
                    self.preconditioner.setLinearizationPoint(z)
                solver.set_preconditioner(self.preconditioner)

            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1

            # Solve by CG  
            solver.solve(zhat, -mg)
            self.total_cg_iter += cost_hessian.ncalls
            
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            
            mg_zhat = mg.inner(zhat)
            
            while descent == 0 and n_backtrack < max_backtracking_iter:
                # Update the step with line search step 
                zstar.zero()
                zstar.axpy(1.0, z)
                zstar.axpy(alpha, zhat)
                cost_new = self.cost_functional.cost(zstar, order=0)
                  
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * mg_zhat) or (-mg_zhat <= self.parameters["gdz_tolerance"]):
                    # conditions are satisfied 
                    cost_old = cost_new
                    descent = 1
                    z.zero()
                    z.axpy(1.0, zstar)

                    # Evaluate cost functional and prepare Hessian 
                    self.cost_functional.cost(z, order=2)
                else:
                    # backtrack 
                    n_backtrack += 1
                    alpha *= 0.5
                            
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15}".format(
                      "It", "cg_it", "cost", "(g,dz)", "||g||L2", "alpha", "tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:15e}".format(
                        self.it, cost_hessian.ncalls, cost_new, mg_zhat, gradnorm, alpha, tolcg) )
                
            if self.callback:
                self.callback(self.it, z)
                
                
            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -mg_zhat <= self.parameters["gdz_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return z
    
    def _solve_tr(self,z):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        cg_max_iter         = self.parameters["cg_max_iter"]
        
        eta_TR = self.parameters["TR"]["eta"]
        delta_TR = None
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        zhat = self.cost_functional.generate_vector(CONTROL) 
        R_zhat = self.cost_functional.generate_vector(CONTROL)   
        zstar = self.cost_functional.generate_vector(CONTROL)
        mg = self.cost_functional.generate_vector(CONTROL)

        
        # cost_old, reg_old, misfit_old = self.model.cost(x)
        cost_old = self.cost_functional.cost(z, order=2)

        while (self.it < max_iter) and (self.converged == False):
            # self.model.solveAdj(x[ADJOINT], x)
            # self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            # gradnorm = self.model.evalGradientParameter(x, mg)

            gradnorm = self.cost_functional.costGrad(mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            cost_hessian = ControlCostHessian(self.cost_functional)
            solver = CGSolverSteihaug()
            solver.set_operator(cost_hessian)

            if self.preconditioner is not None:
                if hasattr(self.preconditioner, 'setLinearizationPoint'):
                    self.preconditioner.setLinearizationPoint(z)
                solver.set_preconditioner(self.preconditioner)

            if self.it > 1:
                solver.set_TR(delta_TR, self.norm_weighting)

            solver.parameters["rel_tolerance"] = tolcg
            self.parameters["max_iter"]        = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(zhat, -mg)
            self.total_cg_iter += cost_hessian.ncalls

            if self.it == 1:
                if self.norm_weighting is not None:
                    self.norm_weighting.mult(zhat,R_zhat)
                    zhat_Rnorm = R_zhat.inner(zhat)
                    delta_TR = max(math.sqrt(zhat_Rnorm),1)
                else:
                    # Use the l2 norm 
                    zhat_Rnorm = zhat.inner(zhat)
                    delta_TR = max(math.sqrt(zhat_Rnorm),1)

            zstar.zero()
            zstar.axpy(1.0, z)
            zstar.axpy(1.0, zhat)
            cost_star = self.cost_functional.cost(zstar, order=0)


            ACTUAL_RED = cost_old - cost_star
            #Calculate Predicted Reduction
            Hzhat = self.cost_functional.generate_vector(CONTROL)
            Hzhat.zero()
            cost_hessian.mult(zhat, Hzhat)
            mg_zhat = mg.inner(zhat)
            PRED_RED = -0.5*zhat.inner(Hzhat) - mg_zhat
            # print( "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED)
            rho_TR = ACTUAL_RED/PRED_RED


            # Nocedal and Wright Trust Region conditions (page 69)
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and solver.reasonid == 3:
                delta_TR *= 2.0
            

            # print( "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n")
            if rho_TR > eta_TR:
                # accept the step and update 
                z.zero()
                z.axpy(1.0, zstar)
                cost_old = cost_star
                accept_step = True

                # Run cost and prepare for Hessian 
                self.cost_functional.cost(z, order=2)
            else:
                accept_step = False
                
            if self.callback:
                self.callback(self.it, z)
                
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:14} {5:14} {6:14} {7:11} {8:14}".format(
                      "It", "cg_it", "cost", "(g,dz)", "||g||L2", "TR Radius", "rho_TR", "Accept Step","tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:14e} {5:14e} {6:14e} {7:11} {8:14e}".format(
                        self.it, cost_hessian.ncalls, cost_old, mg_zhat, gradnorm, delta_TR, rho_TR, accept_step,tolcg) )

            #TR radius can make this term arbitrarily small and prematurely exit.
            if -mg_zhat <= self.parameters["gdz_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old
        return z


