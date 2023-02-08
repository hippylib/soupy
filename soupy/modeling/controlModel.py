import math 

import numpy as np 
import dolfin as dl 
import ufl

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp


from .variables import STATE, PARAMETER, ADJOINT, CONTROL

class ControlModel:
    """
    This class contains the structure needed to evaluate the control objective 
    As inputs it takes a :code:`PDEProblem object`, and a :code:`Qoi` object.
    
    In the following we will denote with
        - :code:`u` the state variable
        - :code:`m` the (model) parameter variable
        - :code:`p` the adjoint variable
        - :code:`z` the control variable
        
    """
    

    def __init__(self, problem, qoi):
        """
        Create a model given:
            - problem: the description of the forward/adjoint problem and all the sensitivities
            - qoi: the qoi component of the cost functional
        """
        self.problem = problem
        self.qoi = qoi
        self.gauss_newton_approx = False
        
        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0
        
                
    def generate_vector(self, component = "ALL"):
        """
        By default, return the list :code:`[u,m,p,z]` where:
        
            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
            - :code:`z` is any object that describes the control variable
        
        If :code:`component = STATE` return only :code:`u`
            
        If :code:`component = PARAMETER` return only :code:`m`
            
        If :code:`component = ADJOINT` return only :code:`p`

        If :code:`component = CONTROL` return only :code:`z`
        """ 
        if component == "ALL":
            x = [self.problem.generate_state(),
                 self.problem.generate_parameter(),
                 self.problem.generate_state(),
                 self.problem.generate_control()]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = self.problem.generate_parameter()
        elif component == ADJOINT:
            x = self.problem.generate_state()
        elif component == CONTROL:
            x = self.problem.generate_control()
            
        return x
    
    def init_parameter(self, m):
        """
        Reshape :code:`m` so that it is compatible with the parameter variable
        """
        self.problem.init_parameter(m)

    def init_control(self, z):
        """
        Reshape :code:`z` so that it is compatible with the control variable
        """
        self.problem.init_control(z)

    def cost(self, x):
        """
        Given the list :code:`x = [u,m,p]` which describes the state, parameter, and
        adjoint variable compute the cost functional as the sum of 
        the qoi functional and the regularization functional.
        
        Return the list [cost functional, regularization functional, qoi functional]
        
        .. note:: :code:`p` is not needed to compute the cost functional
        """
        qoi_cost = self.qoi.cost(x)
        # reg_cost = self.prior.cost(x[PARAMETER])
        # return [qoi_cost+reg_cost, reg_cost, qoi_cost]
        return qoi_cost


    def solveFwd(self, out, x):
        """
        Solve the (possibly non-linear) forward problem.
        
        Parameters:
            - :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
            - :code:`x = [u,m,p]` provides
                1) the parameter variable :code:`m` for the solution of the forward problem
                2) the initial guess :code:`u` if the forward problem is non-linear
        
                .. note:: :code:`p` is not accessed.
        """
        self.n_fwd_solve = self.n_fwd_solve + 1
        self.problem.solveFwd(out, x)

    
    def solveAdj(self, out, x):
        """
        Solve the linear adjoint problem.
        Parameters:
            - :code:`out`: is the solution of the adjoint problem (i.e. the adjoint :code:`p`) (Output parameter)
            - :code:`x = [u, m, p]` provides
                1) the parameter variable :code:`m` for assembling the adjoint operator
                2) the state variable :code:`u` for assembling the adjoint right hand side
                .. note:: :code:`p` is not accessed
        """
        self.n_adj_solve = self.n_adj_solve + 1
        rhs = self.problem.generate_state()
        self.qoi.grad(STATE, x, rhs)
        rhs *= -1.
        self.problem.solveAdj(out, x, rhs)
        # print("RHS", rhs.get_local()[:5])
        # print("ADJSOL", out.get_local()[:5])


    def evalGradientParameter(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u,m,p]`.
        Parameters:
            - :code:`x = [u,m,p]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)`, mtest being a test function in the parameter space \
            (Output parameter)
        
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """ 
        tmp = self.generate_vector(PARAMETER)
        self.problem.evalGradientParameter(x, mg)
        self.qoi.grad(PARAMETER,x,tmp)
        mg.axpy(1., tmp)
        # if not qoi_only:
        #     self.prior.grad(x[PARAMETER], tmp)
        #     mg.axpy(1., tmp)
        
        # self.prior.Msolver.solve(tmp, mg)
        #self.prior.Rsolver.solve(tmp, mg)
        return math.sqrt(mg.inner(tmp))

    
    def evalGradientControl(self,x, mg):
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u,m,p]`.
        Parameters:
            - :code:`x = [u,m,p]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)`, mtest being a test function in the parameter space \
            (Output parameter)
        
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """ 
        tmp = self.generate_vector(CONTROL)
        self.problem.evalGradientControl(x, mg)
        # print("PDE GRAD: ", mg.get_local())
        self.qoi.grad(CONTROL,x,tmp)
        # print("MIDFIT GRAD: ", tmp.get_local())
        mg.axpy(1., tmp)
        # print("OVERALL GRAD: ", mg.get_local())
        return math.sqrt(mg.inner(tmp))

    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point :code:`x = [u,m,p]` at which the Hessian operator (or the Gauss-Newton approximation)
        needs to be evaluated.
        Parameters:
            - :code:`x = [u,m,p]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.
            - :code:`gauss_newton_approx (bool)`: whether to use Gauss-Newton approximation (default: use Newton) 
            
        .. note:: This routine should either:
            - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
            - or partially precompute the block of the hessian (if feasible)
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        self.qoi.setLinearizationPoint(x, self.gauss_newton_approx)
        # if hasattr(self.prior, "setLinearizationPoint"):
        #     self.prior.setLinearizationPoint(x[PARAMETER], self.gauss_newton_approx)

        
    def solveFwdIncremental(self, sol, rhs):
        """
        Solve the linearized (incremental) forward problem for a given right-hand side
        Parameters:
            - :code:`sol` the solution of the linearized forward problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, False)

        
    def solveAdjIncremental(self, sol, rhs):
        """
        Solve the incremental adjoint problem for a given right-hand side
        Parameters:
            - :code:`sol` the solution of the incremental adjoint problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, True)

    
    def applyC(self, dm, out):
        """
        Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
        :code:`out` = :math:`C dm`
        
        Parameters:
            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`C` block on :code:`dm`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)

    def applyCz(self, dz, out):
        """
        Apply the :math:`C` block of the Hessian to a (incremental) control variable, i.e.
        :code:`out` = :math:`C_z dz`
        
        Parameters:
            - :code:`dz` the (incremental) control variable
            - :code:`out` the action of the :math:`C` block on :code:`dm`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT, CONTROL, dz, out)
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C^t dp`
        Parameters:
            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C^T` block on :code:`dp`
            
        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    def applyCzt(self, dp, out):
        """
        Apply the transpose of the :math:`C_z` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C_z^t dp`
        Parameters:
            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C_z^T` block on :code:`dp`
            
        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(CONTROL, ADJOINT, dp, out)
    
    def applyWuu(self, du, out):
        """
        Apply the :math:`W_{uu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{uu} du`
        
        Parameters:
            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{uu}` block on :code:`du`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.qoi.apply_ij(STATE,STATE, du, out)
        if not self.gauss_newton_approx:
            tmp = self.generate_vector(STATE)
            self.problem.apply_ij(STATE,STATE, du, tmp)
            # print("NORM Wuu du", np.linalg.norm(tmp.get_local()))
            out.axpy(1., tmp)
            # print("NORM Wuu du", np.linalg.norm(out.get_local()))

    
    def applyWum(self, dm, out):
        """
        Apply the :math:`W_{um}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{um} dm`
        
        Parameters:
            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{um}` block on :code:`du`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(STATE,PARAMETER, dm, out)
            tmp = self.generate_vector(STATE)
            self.qoi.apply_ij(STATE,PARAMETER, dm, tmp)
            out.axpy(1., tmp)

    
    def applyWuz(self, dz, out):
        """
        Apply the :math:`W_{uz}` block of the Hessian to a (incremental) control variable.
        :code:`out` = :math:`W_{uz} dz`
        
        Parameters:
            - :code:`dz` the (incremental) control variable
            - :code:`out` the action of the :math:`W_{uz}` block on :code:`du`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(STATE, CONTROL, dz, out)
            tmp = self.generate_vector(STATE)
            self.qoi.apply_ij(STATE, CONTROL, dz, tmp)
            out.axpy(1., tmp)
    
    def applyWmu(self, du, out):
        """
        Apply the :math:`W_{mu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{mu} du`
        
        Parameters:
            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{mu}` block on :code:`du`
        
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(PARAMETER, STATE, du, out)
            tmp = self.generate_vector(PARAMETER)
            self.qoi.apply_ij(PARAMETER, STATE, du, tmp)
            out.axpy(1., tmp)

    def applyWzu(self, du, out):
        """
        Apply the :math:`W_{zu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{zu} du`
        
        Parameters:
            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{zu}` block on :code:`du`
        
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(CONTROL, STATE, du, out)
            tmp = self.generate_vector(CONTROL)
            self.qoi.apply_ij(CONTROL, STATE, du, tmp)
            out.axpy(1., tmp)

    
    # def applyR(self, dm, out):
    #     """
    #     Apply the regularization :math:`R` to a (incremental) parameter variable.
    #     :code:`out` = :math:`R dm`
        
    #     Parameters:
    #         - :code:`dm` the (incremental) parameter variable
    #         - :code:`out` the action of :math:`R` on :code:`dm`
        
    #     .. note:: This routine assumes that :code:`out` has the correct shape.
    #     """
    #     self.prior.R.mult(dm, out)

    
    # def Rsolver(self):
    #     """
    #     Return an object :code:`Rsovler` that is a suitable solver for the regularization
    #     operator :math:`R`.
        
    #     The solver object should implement the method :code:`Rsolver.solve(z,r)` such that
    #     :math:`Rz \approx r`.
    #     """
    #     return self.prior.Rsolver

    
    def applyWmm(self, dm, out):
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`
        
        Parameters:
        
            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{mm}` on block :code:`dm`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(PARAMETER,PARAMETER, dm, out)
            tmp = self.generate_vector(PARAMETER)
            self.qoi.apply_ij(PARAMETER,PARAMETER, dm, tmp)
            out.axpy(1., tmp)

    def applyWzz(self, dz, out):
        """
        Apply the :math:`W_{zz}` block of the Hessian to a (incremental) control variable.
        :code:`out` = :math:`W_{zz} dz`
        
        Parameters:
        
            - :code:`dm` the (incremental) control variable
            - :code:`out` the action of the :math:`W_{zz}` on block :code:`dz`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(CONTROL, CONTROL, dz, out)
            tmp = self.generate_vector(CONTROL)
            self.qoi.apply_ij(CONTROL, CONTROL, dz, tmp)
            out.axpy(1., tmp)
            
    def apply_ij(self, i, j, d, out):
        if i == STATE and j == STATE:
            self.applyWuu(d,out)
        elif i == STATE and j == PARAMETER:
            self.applyWum(d,out)
        elif i == STATE and j == CONTROL:
            self.applyWuz(d,out)
        elif i == PARAMETER and j == STATE:
            self.applyWmu(d, out)
        elif i == CONTROL and j == STATE:
            self.applyWzu(d, out)
        elif i == PARAMETER and j == PARAMETER:
            self.applyWmm(d,out)
        elif i == CONTROL and j == CONTROL:
            self.applyWzz(d,out)
        elif i == PARAMETER and j == ADJOINT:
            self.applyCt(d,out)
        elif i == ADJOINT and j == PARAMETER:
            self.applyC(d,out)
        elif i == CONTROL and j == ADJOINT:
            self.applyCzt(d,out)
        elif i == ADJOINT and j == CONTROL:
            self.applyCz(d,out)
        else:
            raise IndexError("apply_ij not allowed for i = {0}, j = {1}".format(i,j))

