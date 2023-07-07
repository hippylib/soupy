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
import ufl

import sys, os
import hippylib as hp


from .variables import STATE, PARAMETER, ADJOINT, CONTROL

class ControlModel:
    """
    This class contains the structure needed to evaluate the control objective 
    As inputs it takes a :py:class:`PDEVariationalControlProblem`, and a :code:`Qoi` object.
    
    In the following we will denote with
     * :code:`u` the state variable
     * :code:`m` the (model) parameter variable
     * :code:`p` the adjoint variable
     * :code:`z` the control variable
    """
    

    def __init__(self, problem, qoi):
        """
        Constructor

        :param problem: The PDE problem 
        :type problem: :py:class:`PDEVariationalControlProblem`
        :param qoi: The quantity of interest
        :type qoi: :py:class:`ControlQoI`
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
         * :code:`u` is any object that describes the state variable
         * :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
         * :code:`p` is any object that describes the adjoint variable
         * :code:`z` is any object that describes the control variable
        
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
        Evaluate the QoI at the a given point :math:`q(u,m,z)`.

        :param x: The point :code:`x = [u,m,p,z]` at which to evaluate the QoI
        :type x: list of :py:class:`dolfin.Vector` objects

        :return: The value of the QoI

        .. note:: :code:`p` is not needed to compute the cost functional
        """
        qoi_cost = self.qoi.cost(x)
        return qoi_cost


    def solveFwd(self, out, x):
        """
        Solve the (possibly non-linear) forward problem.

        :param out: Solution of the forward problem (state)
        :type out: :py:class:`dolfin.Vector`
        :param x: The point :code:`x = [u,m,p,z]`. Provides
            the parameter and control variables :code:`m` 
            and :code:`z` for the solution of the forward problem
            and the initial guess :code:`u` if the forward problem is non-linear
        :type: list of :py:class:`dolfin.Vector` objects

        .. note:: :code:`p` is not accessed.
        """
        self.n_fwd_solve = self.n_fwd_solve + 1
        self.problem.solveFwd(out, x)

    
    def solveAdj(self, out, x):
        """
        Solve the linear adjoint problem.

        :param out: Solution of the forward problem (state)
        :type out: :py:class:`dolfin.Vector`
        :param x: The point :code:`x = [u,m,p,z]`. Provides
            the state, parameter and control variables :code:`u`, :code:`m`,
            and :code:`z` for the solution assembling the adjoint operator.
            Vector :code:`u` is also used to assemble the adjoint right hand side.
        :type: list of :py:class:`dolfin.Vector` objects

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
        Evaluate the :math:`m` gradient action form at the point :math:`(u,m,p,z)`

        :param x: The point :code:`x = [u,m,p,z]`. Provides
            the state, parameter, adjoint, and control variables :code:`u`, :code:`m`, :code:`p`,
            and :code:`z` for the assembling the gradient action.
            Vector :code:`u` is also used to assemble the adjoint right hand side.
        :type: list of :py:class:`dolfin.Vector` objects
        :param mg: Dual of the gradient with respect to the parameter i.e. :math:`(g_m, m_{\mathrm{test}})_{M}`
        :type mg: :py:class:`dolfin.Vector`

        :return: the norm of the gradient in the correct inner product :math:`(g_m,g_m)_{M}^{1/2}`
        """ 
        tmp = self.generate_vector(PARAMETER)
        self.problem.evalGradientParameter(x, mg)
        self.qoi.grad(PARAMETER,x,tmp)
        mg.axpy(1., tmp)
        return math.sqrt(mg.inner(tmp))

    
    def evalGradientControl(self,x, mg):
        """
        Evaluate the :math:`z` gradient action form at the point :math:`(u,m,p,z)`

        :param x: The point :code:`x = [u,m,p,z]`. Provides
            the state, parameter, adjoint, and control variables :code:`u`, :code:`m`, :code:`p`,
            and :code:`z` for the assembling the gradient action
            Vector :code:`u` is also used to assemble the adjoint right hand side.
        :type: list of :py:class:`dolfin.Vector` objects
        :param mg: Dual of the gradient with respect to the control i.e. :math:`(g_z, z_{\mathrm{test}})_{Z}`
        :type mg: :py:class:`dolfin.Vector`

        :return: the norm of the gradient in the correct inner product :math:`(g_z,g_z)_{Z}^{1/2}`
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
        Specify the point :code:`x = [u,m,p,z]` at which the Hessian operator 
            (or the Gauss-Newton approximation) needs to be evaluated.

        :param x: The point :code:`x = [u,m,p,z]` for which the Hessian needs to be evaluated
        :type: list of :py:class:`dolfin.Vector` objects
        :param gauss_newton_approx: whether to use the Gauss-Newton approximation (default: use Newton)
        :type gauss_newton_approx: bool
            
        .. note:: This routine should either:
         1. simply store a copy of x and evaluate action of blocks of the Hessian on the fly, or
         2. partially precompute the block of the hessian (if feasible)
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        self.qoi.setLinearizationPoint(x, self.gauss_newton_approx)
        # if hasattr(self.prior, "setLinearizationPoint"):
        #     self.prior.setLinearizationPoint(x[PARAMETER], self.gauss_newton_approx)

        
    def solveFwdIncremental(self, sol, rhs):
        """
        Solve the linearized (incremental) forward problem for a given right-hand side

        :param sol: Solution of the incremental forward problem (state)
        :type sol: :py:class:`dolfin.Vector`
        :param rhs: Right hand side of the linear system
        :type rhs: :py:class:`dolfin.Vector`
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, False)

        
    def solveAdjIncremental(self, sol, rhs):
        """
        Solve the incremental adjoint problem for a given right-hand side

        :param sol: Solution of the incremental adjoint problem (adjoint)
        :type sol: :py:class:`dolfin.Vector`
        :param rhs: Right hand side of the linear system
        :type rhs: :py:class:`dolfin.Vector`
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, True)

    
    def applyC(self, dm, out):
        """
        Apply the :math:`C_{m}` block of the Hessian to a (incremental) parameter variable, i.e.
        :code:`out` = :math:`C_{m} dm`
        
        :param dm: The (incremental) parameter variable
        :type dm: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`C_z` block on :code:`dm`
        :type out: :py:class:`dolfin.Vector`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)

    def applyCz(self, dz, out):
        """
        Apply the :math:`C_z` block of the Hessian to a (incremental) control variable, i.e.
        :code:`out` = :math:`C_z dz`
        
        :param dz: The (incremental) control variable
        :type dz: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`C_z` block on :code:`dz`
        :type out: :py:class:`dolfin.Vector`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT, CONTROL, dz, out)
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the :math:`C_{m}` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C_{m}^T dp`

        :param dp: The (incremental) adjoint variable
        :type dp: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`C_{m}^T` block on :code:`dp`
        :type out: :py:class:`dolfin.Vector`

        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    def applyCzt(self, dp, out):
        """
        Apply the transpose of the :math:`C_{z}` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C_{z}^T dp`

        :param dp: The (incremental) adjoint variable
        :type dp: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`C_{z}^T` block on :code:`dp`
        :type out: :py:class:`dolfin.Vector`

        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(CONTROL, ADJOINT, dp, out)
    
    def applyWuu(self, du, out):
        """
        Apply the :math:`W_{uu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{uu} du`

        :param du: The (incremental) state variable
        :type du: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{uu}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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

        :param dm: The (incremental) parameter variable
        :type dm: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{um}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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
        
        :param dz: The (incremental) control variable
        :type dz: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{uz}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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

        :param du: The (incremental) state variable
        :type du: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{mu}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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
        
        :param du: The (incremental) state variable
        :type du: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{zu}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(CONTROL, STATE, du, out)
            tmp = self.generate_vector(CONTROL)
            self.qoi.apply_ij(CONTROL, STATE, du, tmp)
            out.axpy(1., tmp)

    
    def applyWmm(self, dm, out):
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`
        
        :param dm: The (incremental) parameter variable
        :type dm: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{mm}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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
        
        :param dz: The (incremental) control variable
        :type dz: :py:class:`dolfin.Vector`
        :param out: The action of the :math:`W_{zz}` block on :code:`du`
        :type out: :py:class:`dolfin.Vector`

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
        """
        Apply the :math:`(i,j)` block of the Hessian to a vector :code:`d`

        :param i: The output variable index (0, 1, 2, 3)
        :type i: int
        :param j: The input variable index (0, 1, 2, 3)
        :type j: int
        :param d: The vector to which the Hessian is applied
        :type d: :py:class:`dolfin.Vector`
        :param out: The action of the Hessian on :code:`d`
        :type out: :py:class:`dolfin.Vector`
        """
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

