import sys
import math
import time 

import dolfin as dl 
import numpy as np 
from mpi4py import MPI 

import hippylib as hp 

def applyBC(bcs, u):
    """
    Applies a single or a list of :code:`dolfin.DirichletBC` 
    to the vector :code:`u`
    """
    if bcs is None:
        pass
    elif isinstance(bcs, list):
        for bc in bcs:
            bc.apply(u)
    else:
        bcs.apply(u)

def homogenizeBC(bcs):
    """
    Converts a single or a list of :code:`dolfin.DirichletBC` 
    to their homogenized (zeroed) forms
    """
    if bcs is None:
        return None
    elif isinstance(bcs, list):
        bcs0 = []
        for bc in bcs:
            bc0 = dl.DirichletBC(bc)
            bc0.homogenize()
            bcs0.append(bc0)
        return bcs0
    else:
        bc0 = dl.DirichletBC(bcs)
        bc0.homogenize()
        return bc0 

class NewtonBacktrackSolver:
    """
    Backtracking Newton solver for the nonlinear variational system 

        .. math:: r(u,v) = 0 \\forall v

    The user must provide the variational forms for the residual form, \
        an initial guess, boundary conditions, and optionally, \
        variational forms for the Jacobian and backtracking cost functional.
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance"       #3
                           ]
    
    def __init__(self):
        """
        Initializes the :code:`NewtonBacktrackSolver` with the following parameters.

         - :code:`rel_tolerance`         --> we converge when :math:`\|r\|_2/\|r_0\|_2 \leq` :code:`rel_tolerance`
         - :code:`abs_tolerance`         --> we converge when :math:`\|r\|_2 \leq` :code:`abs_tolerance`
         - :code:`maxximum_iterations`   --> maximum number of iterations
         - :code:`c_armijo`              --> Armijo constant for sufficient reduction
         - :code:`max_backtracking_iter` --> Maximum number of backtracking iterations
         - :code:`print_level`           --> Print info on screen
        """        
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-8
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["maximum_iterations"]    = 20
        self.parameters["c_armijo"]              = 1e-4
        self.parameters["max_backtracking_iter"] = 20
        self.parameters["print_level"]           = 1
        self.parameters["lu_method"]             = "mumps"
        
    def solve(self, residual_form, u, bcs=None, J_form = None, energy_form=None, solver=None):
        """
        Solve the nonlinear variational system 

        .. math:: r(u,v) = 0 \\forall v

        given using a backtracking Newton method with supplied initial guess 

        :param residual_form: Variational form for the residual 
        :param u: Initial guess for the solution and funciton used in :code:`residual_form`
        :type u: :py:class:`dolfin.Function`
        :param bcs: List of boundary conditions 
        :type bcs: list 
        :param J_form: Variational form for the Jacobian of the residual 
        :param energy_form: Optional variational form for the energy functional. 
            If supplied, uses this as the backtracking cost
        :param solver: Optional linear solver with method :code:`solve(A, x, b)`
            Will initialize a solver if :code:`solver` is :code:`None`.
        """

        self.it = 0
        self.converged = False
        self.reason = 0
        self.final_grad_norm = 0

        if J_form is None:
            J_form = dl.derivative(residual_form, u)
            
        mpi_comm = u.vector().mpi_comm()
        rtol = self.parameters["rel_tolerance"]
        atol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["maximum_iterations"]
        c_armijo = self.parameters["c_armijo"] 
        max_backtrack = self.parameters["max_backtracking_iter"]
        print_level =  self.parameters["print_level"]
        lu_method = self.parameters["lu_method"]

        my_rank = MPI.COMM_WORLD.Get_rank()

        # Apply BC to initial guess
        applyBC(bcs, u.vector()) 

        # Homogenize the dirichlet BCs if they exist 
        bcs0 = homogenizeBC(bcs)

        # Assemble initial residual
        residual = dl.assemble(residual_form)
        # applyBC(bcs0, residual)
        r_norm = residual.norm("l2")
        tol = max(r_norm*rtol, atol)
        if energy_form is not None:
            energy = dl.assemble(energy_form)

        du = dl.Vector(u.vector())
        u_current = dl.Vector(u.vector())

        if print_level > 0 and my_rank == 0:
            if energy_form is not None:
                print("{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} ".format(
                    "MPI", "It", "Energy", "||r||", "(r,du)", "alpha"))

                print("{0:3d} {1:3d} {2:15f} {3:15f} {4:15f} {5:15f}".format(
                    my_rank, self.it, energy, r_norm, 0, 0))               
            else:
                print("{0:3} {1:3} {2:15} {3:15} {4:15}".format(
                    "MPI", "It", "||r||", "(r,du)", "alpha"))

                print("{0:3d} {1:3d} {2:15f} {3:15f} {4:15f}".format(
                    my_rank, self.it, r_norm, 0, 0))
        
        if solver is None:
            solver = hp.PETScLUSolver(mpi_comm, method=lu_method)

        sys.stdout.flush()
        while self.it < max_iter:
            # Make Newton system 
            J, r = dl.assemble_system(J_form, residual_form, bcs=bcs0)
            solver.set_operator(J)

            # Solving the newton system. Note -du is the descent direction
            tsolve0 = time.time()
            solver.solve(du, r)
            tsolve1 = time.time()
            if print_level > 0 and my_rank == 0:
                print("Proc %d Linear solve time = %.3g s" %(my_rank, tsolve1 - tsolve0))
            sys.stdout.flush()

            du_rn = residual.inner(du)

            i_backtrack = 0 
            backtrack_converged = False
            alpha = 1.0
            u_current.zero() 
            u_current.axpy(1.0, u.vector())

            while not backtrack_converged and i_backtrack < max_backtrack:
                u.vector().zero()
                u.vector().axpy(1.0, u_current)
                u.vector().axpy(-alpha, du)

                if energy_form is not None:
                    # Backtrack on the energy
                    energy_new = dl.assemble(energy_form)
                    if energy_new < energy:
                    # if energy_new < energy + alpha * c_armijo * du_rn:
                        backtrack_converged = True
                        energy = energy_new
                    else:
                        alpha = alpha/2
                
                else:
                    # Backtrack on residual norm 
                    residual_new = dl.assemble(residual_form)
                    applyBC(bcs0, residual_new)
                    r_new_norm = residual_new.norm('l2')
                    if r_new_norm < r_norm:
                        backtrack_converged = True
                        r_norm = r_new_norm
                    else:
                        alpha = alpha/2

                i_backtrack += 1  

            # Done with backtracking attempts
            if not backtrack_converged:
                self.reason = 2
                u.vector().zero()
                u.vector().axpy(1.0, u_current)
                break

            # Back tracking is successful, increment iteration
            self.it += 1        

            # Compute residual norm 
            residual = dl.assemble(residual_form)
            applyBC(bcs0, residual)
            r_norm = residual.norm("l2")

            # Print new residual norms
            if print_level > 0 and my_rank == 0:
                if energy_form is not None:
                    print("{0:3d} {1:3d} {2:15f} {3:15f} {4:15f} {5:15f}".format(
                        my_rank, self.it, energy, r_norm, du_rn, alpha))
                else:
                    print("{0:3d} {1:3d} {2:15f} {3:15f} {4:15f}".format(
                        my_rank, self.it, r_norm, du_rn, alpha))

            if r_norm < tol:
                self.converged = True
                self.reason = 1
                break

            sys.stdout.flush()

        # Finished optimization loop             
        self.final_residual_norm = r_norm
        if print_level > -1 and my_rank == 0:
            print(self.termination_reasons[self.reason])
            if self.converged:
                print("Proc %d: Backtracking Newton converged after %d iterations" %(my_rank, self.it))
            else:
                print("Proc %d: Backtracking Newton did not converge after %d iterations" %(my_rank, self.it))

            print("Proc %d: Final norm of the residual: %.5e" %(my_rank, r_norm))
            if energy_form is not None:
                print("Proc %d: Final value of energy: %.5e" %(my_rank, energy))
                
        sys.stdout.flush()
        return self.it, self.converged
