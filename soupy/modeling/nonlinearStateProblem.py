'''
Created on Feb 18, 2016

@author: uvilla
'''
import dolfin as dl
import numpy as np
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 

class NonlinearResidualHandler:
    def residual(self, u, u_test, m, z):
        """
        returns the residual form 
        """
        raise NotImplementedError("Child class should implement residual")

    def Jacobian(self, u, u_test, m, z):
        raise NotImplementedError("Child class should implement Jacobian")

    def mass(self, u, u_test):
        """
        returns the mass matrix 
        """
        raise NotImplementedError("Child class should implement mass")

    def stiffness(self, u, u_test):
        """
        returns the stiffness matrix 
        """
        raise NotImplementedError("Child class should implement stiffness")



class NonlinearStateProblem:
    """
    This class provides methods to assemble the jacobian and residual of a steady-state nonlinear problem.
    It can be passed to any of the non-linear solvers implemented in hippylib.
    """
    def __init__(self, Vh_STATE, residual_handler, bcs, bcs0):
        """
        Constructor:
        - Vh_STATE: the finite element space for the state
        - residual_handler: a residual_handler object that provides the methods:
                 - residual: return the residual weak form
                 - Jacobian: return the Jacobian weak form
                 - mass: return the mass weak form
                 - stiffness: return the stiffness weak form
        - bcs: list of Dirichlet B.C. for the forward problem.
        - bcs0: list of homogeneus Dirichlet B.C. for the adjoint problem
        """
        self.Vh_STATE = Vh_STATE
        self.residual_handler = residual_handler
        self.x_trial = dl.TrialFunction(Vh_STATE)
        self.x_test = dl.TestFunction(Vh_STATE)
        self.bcs = bcs
        self.bcs0 = bcs0
        self.fcp = {}
        self.fcp["quadrature_degree"] = 6
        
        self.xdummy = dl.Function(Vh_STATE)        
        self.dummyform = dl.inner( self.xdummy, self.x_test)*dl.dx

        
    def applyBC(self,x):
        """
        Apply the boundary conditions.
        """
        [bc.apply(x) for bc in self.bcs]
        
    def applyBC0(self,r):
        """
        Apply the homogeneus boundary conditions.
        """
        [bc.apply(r) for bc in self.bcs0]
        
    def Norm(self, coeffsL2=None, coeffsH1=None, apply_bc=False):
        """
        Returns a finite element matrix for a (weighted) H^1 norm.
        INPUTS:
        - coeffsL2: weights for the L2 norm (DEFAULT: 1)
        - coeffsH1: weights for the H1 norm (DEFAULT: 1)
        - apply_bc: if True apply essential b.c. (DEFAULT: False) 
        """
        Wform = self.residual_handler.mass(self.x_trial, self.x_test, coeffsL2) \
                +self.residual_handler.stiffness(self.x_trial, self.x_test, coeffsH1)
        
        if apply_bc:
            W, dummy = dl.assemble_system(Wform, self.dummyform, self.bcs0, form_compiler_parameters=self.fcp)
        else:
            W =  dl.assemble(Wform, form_compiler_parameters=self.fcp)
            
        return W
                            
    def residual(self, x, extra_args=()):
        """
        Evaluate the residual at point x.
        extra_args is a tuple of additional arguments necessary to assemble the residual
        """
        xfun = hp.vector2Function(x, self.Vh_STATE)

        rform = self.residual_handler.residual(xfun, self.x_test, *extra_args)
        r = dl.assemble(rform, form_compiler_parameters=self.fcp)
        
        self.applyBC0(r)
        return r
    
    def Jacobian(self, x, extra_args=()):
        """
        Assemble the Jacobian operator at point x.
        extra_args is a tuple of additional arguments necessary to assemble the Jacobian
        """
        xfun = hp.vector2Function(x, self.Vh_STATE)

        Jform = self.residual_handler.Jacobian(xfun, self.x_test, self.x_trial, *extra_args)
                                    
        J, dummy = dl.assemble_system(Jform, self.dummyform, self.bcs0, form_compiler_parameters=self.fcp)
        
        return J
    
    def check_jacobian(self, x0, d, extra_args=()):
        """
        Perform a finite difference check of the Jacobian operator.
        
        INPUTS:
        - x0: the point at which to check the Jacobian
        - d : the direction in which to check the Jacobian
        - extra_args": a tuple of additional arguments necessary to assemble the Jacobian
        """
        
        self.applyBC0(d)
        
        x_plus = dl.Vector(x0.mpi_comm())

        r0 = self.residual(x0, extra_args)
        J0 = self.Jacobian(x0, extra_args )
        J0.init_vector(x_plus, 0)
        J0d = J0*d
        
        eps = np.power(2.0, np.arange(-32,0,1))
        out = np.zeros((eps.shape[0],3))
        
        for i in np.arange(eps.shape[0]):
            my_eps = eps[i]
            x_plus.zero()
            x_plus.axpy(1., x0)
            x_plus.axpy(my_eps, d)
            r_plus = self.residual(x_plus, extra_args)
            
            r_plus.axpy(-1., r0)
            r_plus *= (1./my_eps)
            out[i,2] = r_plus.norm("l2")
            r_plus.axpy(-1., J0d)
            out[i,0] = my_eps
            out[i,1] = r_plus.norm("l2")
            
        return out
