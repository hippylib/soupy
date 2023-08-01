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
import numpy as np
import dolfin as dl
import sys
import os

import hippylib as hp
from ..modeling.variables import CONTROL


def BFGSoperator_ParameterList():
    parameters = {}
    parameters["BFGS_damping"] = [0.2, "Damping of BFGS"]
    parameters["memory_limit"] = [np.inf, "Number of vectors to store in limited memory BFGS"]
    return hp.ParameterList(parameters)

def BFGS_ParameterList():
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-12, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [500, "maximum number of iterations"]
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [25, "Maximum number of backtracking iterations"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["BFGS_op"]               = [BFGSoperator_ParameterList(), "BFGS operator"]
    return hp.ParameterList(parameters)

class RescaledIdentity(object):
    """
    Default operator for :code:`H0inv`, corresponds to applying :math:`d0 I`
    """
    def __init__(self, init_vector=None):
        self.d0 = 1.0
        self._init_vector = init_vector
        
    def init_vector(self, x, dim):
        if self._init_vector:
            self._init_vector(x,dim)
        else:
            raise

    def solve(self, x, b):
        """
        Applies the operator :math:`d0 I` 

        :param x: solution vector
        :type x: :py:class:`dolfin.Vector` 
        :param b: right-hand side vector
        :type b: :py:class:`dolfin.Vector` 
        """

        # print("\t\t WITHIN IDENTITY")
        # print("\t\t", self.d0)
        # print("\t\t", x, x.get_local())
        # print("\t\t", b, b.get_local())
        x.zero()
        x.axpy(self.d0, b)




class BFGS_operator:

    def __init__(self,  parameters=BFGSoperator_ParameterList()):
        self.S, self.Y, self.R = [],[],[]

        self.H0inv = None
        self.help = None
        self.help2 = None
        self.update_scaling = True

        self.parameters = parameters

    def set_H0inv(self, H0inv):
        """
        Set user-defined operator corresponding to :code:`H0inv`

        :param H0inv: Fenics operator with method :code:`solve()`
        """
        self.H0inv = H0inv
        
    def solve(self, x, b):
        """
        Solve system:           
        :math:`H_{\mathrm{bfgs}} x = b`
        where :math:`H_{\mathrm{bfgs}}` is the approximation to the Hessian build by BFGS. 
        That is, we apply :math:`x = (H_{\mathrm{bfgs}})^{-1} b = H_k b`
        where :math:`H_k` matrix is BFGS approximation to the inverse of the Hessian.
        Computation done via double-loop algorithm.
        
        :param x: The solution to the system
        :type x: :py:class:`dolfin.Vector` 
        :param b: The right-hand side of the system
        :type b: :py:class:`dolfin.Vector` 
        """
        A = []
        if self.help is None:
            self.help = b.copy()
        else:
            # print("\t WITHIN BFGSop solve, before assignment")
            # print("\t", x, x.get_local())
            # print("\t", self.help, self.help.get_local())
            self.help.zero()
            self.help.axpy(1.0, b)

        for s, y, r in zip(reversed(self.S), reversed(self.Y), reversed(self.R)):
            a = r * s.inner(self.help)
            A.append(a)
            self.help.axpy(-a, y)
        
        # print("\t WITHIN BFGSop solve, near solve")
        # print("\t", x, x.get_local())
        # print("\t", self.help, self.help.get_local())
        self.H0inv.solve(x, self.help)     # x = H0 * x_copy

        for s, y, r, a in zip(self.S, self.Y, self.R, reversed(A)):
            b = r * y.inner(x)
            x.axpy(a - b, s)


    def update(self, s, y):
        """
        Update BFGS operator with most recent gradient update.
        To handle potential break from secant condition, update done via damping

        :param s: The update in medium parameters
        :type s: :py:class:`dolfin.Vector` 
        :param y: The update in gradient 
        :type y: :py:class:`dolfin.Vector` 
        """
        damp = self.parameters["BFGS_damping"]
        memlim = self.parameters["memory_limit"]
        if self.help2 is None:
            self.help2 = y.copy()
        else:
            self.help2.zero()

        sy = s.inner(y)
        # print("\t Within update")
        # print("\t", y, y.get_local())
        # print("\t", self.help2, self.help2.get_local())
        self.solve(self.help2, y)
        yHy = y.inner(self.help)
        theta = 1.0
        if sy < damp*yHy:
            theta = (1.0-damp)*yHy/(yHy-sy)
            s *= theta
            s.axpy(1-theta, self.help2)
            sy = s.inner(y)
        assert(sy > 0.)
        rho = 1./sy
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.R.append(rho)

        # if L-BFGS
        if len(self.S) > memlim:
            self.S.pop(0)
            self.Y.pop(0)
            self.R.pop(0)
            self.update_scaling = True

        # re-scale H0 based on earliest secant information
        if hasattr(self.H0inv, "d0") and self.update_scaling:
            s0  = self.S[0]
            y0 = self.Y[0]
            d0 = s0.inner(y0) / y0.inner(y0)
            self.H0inv.d0 = d0
            self.update_scaling = False

        return theta





class BFGS:
    """
    Implement BFGS technique with backtracking inexact line search and damped updating \
        See `Nocedal & Wright (06), ch.6.2, ch.7.3, ch.18.3` \
        The user must provide a :py:class:`ControlCostFunctional` that describes the forward problem, \
        cost functional, and all the \
        derivatives for the gradient and the Hessian.
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance"       #3
                           ]

    def __init__(self, cost_functional, parameters=BFGS_ParameterList()):
        """
        Initialize the BFGS solver.
        Type :code:`BFGS_ParameterList().showMe()` for default parameters and their description

        :param cost_functional: The cost functional
        :param parameters: The parameters for the BFGS solver
        """
        self.cost_functional = cost_functional
        
        self.parameters = parameters        
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0

        self.BFGSop = BFGS_operator(self.parameters["BFGS_op"])


    def solve(self, z, H0inv=RescaledIdentity(), box_bounds=None, constraint_projection=None):
        """
        Solve the constrained optimization problem with initial guess :code:`z`

        :param z: The initial guess
        :type z: :py:class:`dolfin.Vector` 
        :param H0inv: Initial approximation of the inverse of the Hessian. 
            Has optional method :code:`update(x)` that will update the operator
        :param box_bounds: Bound constraint. A list with two entries (min and max). 
            Can be either a scalar value or a :code:`dolfin.Vector` of the same size as :code:`z`
        :type box_bounds: list 
        :param constraint_projection: Alternative projectable constraint
        :type constraint_projection: :py:class:`ProjectableConstraint`

        :return: The optimization solution :code:`z` and a dictionary of information

        .. note:: The input :code:`z` will be overwritten 
        """
        
        if box_bounds is not None:
            if hasattr(box_bounds[0], "get_local"):
                param_min = box_bounds[0].get_local()    #Assume it is a dolfin vector
            else:
                param_min = box_bounds[0]*np.ones_like(z.get_local()) #Assume it is a scalar
            if hasattr(box_bounds[1], "get_local"):
                param_max = box_bounds[1].get_local()    #Assume it is a dolfin vector
            else:
                param_max = box_bounds[1]*np.ones_like(z.get_local()) #Assume it is a scalar
        
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        # ls_list = self.parameters[self.parameters["globalization"]]
        c_armijo = self.parameters["c_armijo"]
        max_backtracking_iter = self.parameters["max_backtracking_iter"]
        print_level = self.parameters["print_level"]

        self.BFGSop.parameters["BFGS_damping"] = self.parameters["BFGS_op"]["BFGS_damping"]
        self.BFGSop.parameters["memory_limit"] = self.parameters["BFGS_op"]["memory_limit"]
        self.BFGSop.set_H0inv(H0inv)

        # Initialize vectors
        zhat = self.cost_functional.generate_vector(CONTROL)
        g = self.cost_functional.generate_vector(CONTROL)
        dz = self.cost_functional.generate_vector(CONTROL)
        z_star = self.cost_functional.generate_vector(CONTROL)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        costs = [] 
        cost_old = self.cost_functional.cost(z, order=1)
        costs.append(cost_old)

        gradnorms = [] 
        n_backtracks = [] 

        if print_level >= 0:
            print( "\n{0:3} {1:15} {2:15} {3:15} {4:15}".format(
                  "It", "cost", "||g||L2", "||dz||L2", "alpha") )
            print( "{0:3d} {1:15e}".format(
                    self.it, cost_old))
        

        while (self.it < max_iter) and (self.converged == False):
            if hasattr(self.BFGSop.H0inv, "setPoint"):
                self.BFGSop.H0inv.setPoint(z)

            g_old = g.copy()            
            gradnorm = self.cost_functional.costGrad(g)
            if gradnorm is None:
                gradnorm = np.sqrt(g.inner(g))
            gradnorms.append(gradnorm)

            # Update BFGS
            if self.it > 0:
                s = zhat * alpha
                y = g - g_old
                theta = self.BFGSop.update(s, y)
            else:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                print("Tolerance: ", tol)
                theta = 1.0
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1

            self.BFGSop.solve(zhat, -g)
            
            # backtracking line-search
            alpha = 1.0
            descent = False
            n_backtrack = 0
            g_zhat = g.inner(zhat)

            while not descent and n_backtrack < max_backtracking_iter:
                # Update the optimization variable 
                z_star.zero()
                z_star.axpy(1.0, z)
                z_star.axpy(alpha, zhat)
                if box_bounds is not None:
                    z_star.set_local(np.maximum(z_star.get_local(), param_min))
                    z_star.set_local(np.minimum(z_star.get_local(), param_max))
                if constraint_projection is not None:
                    constraint_projection.project(z_star)

                # compute the cost 
                cost_new = self.cost_functional.cost(z_star, order=0)
                if print_level >= 1:
                    print("\tBacktracking cost for step size %g: \t%g" %(alpha, cost_new))
                
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha * c_armijo * g_zhat) or (-g_zhat <= self.parameters["gdm_tolerance"]):
                    descent = True
                else:
                    n_backtrack += 1
                    alpha *= 0.5

            n_backtracks.append(n_backtrack)

            # Compute size of step
            dz.zero()
            dz.axpy(1.0, z_star)
            dz.axpy(-1.0, z)
            dz_norm = np.sqrt(dz.inner(dz))
            
            # Update z and compute cost 
            z.zero()
            z.axpy(1.0, z_star)
            cost_old = self.cost_functional.cost(z, order=1)
            costs.append(cost_old)

            if print_level >= 0:
                print( "{:3d} {:15e} {:15e} {:14e} {:14e} {:14e}".format(
                self.it, cost_old, dz_norm, gradnorm, alpha, theta))

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            if -g_zhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

        result = dict()
        result["costs"] = costs
        result["gradnorms"] = gradnorms
        result["n_backtracks"] = n_backtracks
        result["termination_reason"] = BFGS.termination_reasons[self.reason]
        
        if print_level >= 0:
            print(BFGS.termination_reasons[self.reason])
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old

        return z, result

