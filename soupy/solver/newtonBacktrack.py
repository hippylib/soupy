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

import time 
import dolfin as dl
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys, os
import hippylib as hp 

class NewtonBacktrack:
    """
    Newton-type non linear solver with backtracking line-search.
    Parameters for the nonlinear solver are set through the attribute parameter.
    - "rel_tolerance": the relative tolerance (1e-6 default)
    - "abs_tolerance": the absolute tolerance (1e-12 default)
    - "max_iter": the maximum number of iterations (200 default)
    - "print_level": controls verbosity of the solver to screen
                     -1: no output to screen
                      0: final residual, and convergence reason
                      1: print current residual and alpha at each iteration
    - "max_backtracking": maximum number of backtracking iterations
    """
    def __init__(self, Wr=None, callback=None):
        """
        Constructor
        
        INPUT:
        - state_problem: an object of type NonlinearStateProblem. It provides the residual and the Jacobian
        - Wr: an s.p.d. operator used to compute a weighted norm.
            This object must provide the methods:
            Wr.norm(r):= sqrt( r, Wr*r) to compute the weighted norm of r
            Wr.inner(r1,r2) := (r1, Wr*r2) to compute the weighted inner product of r1 and r2
            If Wr is None the standard l2-norm is used
        - callback: a function handler to perform additional postprocessing (such as output to paraview)
                    of the current solution
        - extra_args: a tuple of additional parameters to evaluate the residual and Jacobian
        """   
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-8
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["max_iter"]              = 200
        self.parameters["print_level"]           = 0
        self.parameters["max_backtracking"]      = 20
        
        self.Wr = Wr
        self.callback = callback

        self.final_it = 0
        self.final_norm = 0
        self.initial_norm = 0
        self.converged = False
        self.linear_solver = "fenics"


    def set_rel_tolerance(self, tol):
        self.parameters["rel_tolerance"] = tol 

    def set_abs_tolerance(self, tol):
        self.parameters["abs_tolerance"] = tol 
                            
    def _norm_r(self,r):
        if self.Wr is None:
            return r.norm("l2")
        else:
            return self.Wr.norm(r)
    
    def _inner_r(self,r1,r2):
        if self.Wr is None:
            return r1.inner(r2)
        else:
            return self.Wr.inner(r1,r2)
        
    def solve(self, problem, x_out, x_in, extra_args=()):
        """
        Solve the nonlinear problem and assign output 
        - :code: `problem` a NonlinearStateProblem object
        - :code: `x_out` output state vector 
        - :code: `x_in` input state vector
        - :code: `extra_args` additional arguments needed to form the residual 
        """
        x = dl.Vector(x_in) # copy the input vector 
        rtol = self.parameters["rel_tolerance"]
        atol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        max_backtracking = self.parameters["max_backtracking"]
        
        problem.applyBC(x)
        
        if self.callback is not None:
            self.callback(x)
        
        r = problem.residual(x, extra_args)
        
        norm_r0 = self._norm_r(r)
        norm_r = norm_r0
        
        self.initial_norm = norm_r0
        
        tol = max(atol, rtol*norm_r0)
        
        self.converged = False
        it = 0
                
        d = dl.Vector()
                
        if(self.parameters["print_level"] >= 1):
                print("\n{0:3} {1:15} {2:15}".format(
                      "It", "||r||", "alpha"))
                
        if self.parameters["print_level"] >= 1:
                print("{0:3d} {1:15e} {2:15e}".format(
                        it, norm_r0, 1.))
        
        while it < max_iter and not self.converged:
            
            t0 = time.time()
            J = problem.Jacobian(x, extra_args)
            # JSolver = dl.PETScKrylovSolver('gmres')
            # print "preconditioner", JSolver.preconditioners()
            # JSolver.absolute_tolerance = 1e-16
            # JSolver.relative_tolerance = 1e-16
            # JSolver.maximum_iterations = 100
            t0 = time.time()
            if self.linear_solver == "fenics":
                JSolver = hp.PETScLUSolver(x.mpi_comm())
                JSolver.set_operator(J)
                JSolver.solve(d, -r)
            else:
                J_mat = dl.as_backend_type(J).mat()
                row,col,val = J_mat.getValuesCSR() # I think they only give the csr, so we convert
                J_csc = sp.csr_matrix((val,col,row)).tocsc()
                J_lu = spla.splu(J_csc)
                d.set_local(J_lu.solve(-r.get_local()))

            if(self.parameters["print_level"] >= 1):
                t1 = time.time()
                print("Jacobian solve took %g s using %s solver" %(t1-t0, self.linear_solver))
            



            
            alpha = 1.
            backtracking_converged = False
            j = 0
            while j < max_backtracking and not backtracking_converged:
                x_new = x + alpha*d
                r = problem.residual(x_new, extra_args)
                try:
                    norm_r_new = self._norm_r(r)
                except:
                    norm_r_new = norm_r+1.
                if norm_r_new  <= norm_r:
                    x.axpy(alpha, d)
                    norm_r = norm_r_new
                    backtracking_converged = True
                else:
                    alpha = .5*alpha
                    j = j+1
                                        
            if not backtracking_converged:
                if self.parameters["print_level"] >= 0:
                    print("Backtracking failed at iteration", it, ". Residual norm is ", norm_r)
                self.converged = False
                self.final_it = it
                self.final_norm = norm_r
                break
                
                
            if norm_r_new < tol:
                if self.parameters["print_level"] >= 0:
                    print("Converged in ", it, "iterations with final residual norm", norm_r_new)
                self.final_norm = norm_r_new
                self.converged = True
                self.final_it = it
                break
                            
            it = it+1
                                
            if self.parameters["print_level"] >= 1:
                print("{0:3d} {1:15e} {2:15e}".format(
                        it,  norm_r, alpha))
    
            if self.callback is not None:
                self.callback(x)


        # Assign result to output vector  
        x_out.zero()
        x_out.axpy(1.0, x)
        
        if not self.converged:
            self.final_norm = norm_r_new
            self.final_it = it
            if self.parameters["print_level"] >= 0:
                    print("Not Converged in ", it, "iterations. Final residual norm", norm_r_new)

        return self.final_it, self.converged
