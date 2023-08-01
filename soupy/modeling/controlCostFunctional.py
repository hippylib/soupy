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

import logging 
import numpy as np
import dolfin as dl

from .variables import STATE, PARAMETER, ADJOINT, CONTROL

class ControlCostFunctional:
    """
    Base class for the cost function for solving an optimal control problem
    under uncertainty.
    """

    def cost(self, z, order=0):
        """
        Given the control variable z evaluate the cost functional. Order specifies \
            the order of derivatives required after the computation of the cost 
        """
        raise NotImplementedError("Child class should implement method costValue")

    def costGrad(self, g):
        """
        Evaluate the gradient of the cost functional. Assumes :code:`cost` is called \
            with order >=1 
        """
        raise NotImplementedError("Child class should implement method costGrad")

    def costHessian(self, zhat, Hzhat):
        """
        Evaluate the Hessian of the cost functional acting in direction :code:`zhat`. \
                Assumes :code:`cost` is called with order >=2 
        """
        raise NotImplementedError("Child class should implement method costHessian")


class DeterministicControlCostFunctional(ControlCostFunctional):
    """
    This class implements a deterministic approximation for the optimal control problem
    under uncertainty by considering a fixed parameter at the mean of the prior

        .. math:: J(z) := Q(\\bar{m}, z) + P(z) 

    """

    def __init__(self, model, prior, penalization=None):
        """
        Constructor

        :param model: control model containing the :code:`soupy.PDEVariationalControlProblem`
            and :code:`soupy.ControlQoI`
        :type model: :py:class:`soupy.ControlModel`
        :param prior: The prior distribution for the random parameter
        :type prior: :py:class:`hippylib.Prior`
        :param penalization: An optional penalization object for the cost of control
        :type penalization: :py:class:`soupy.Penalization`
        """
        self.model = model
        self.prior = prior
        self.penalization = penalization

        self.u, self.m, self.p, self.z = self.model.generate_vector(component="ALL")
        self.m.zero()
        self.m.axpy(1.0, self.prior.mean)
        self.x = [self.u, self.m, self.p, self.z]
        self.diff_helper = self.model.generate_vector(CONTROL)
        self.grad_penalization = self.model.generate_vector(CONTROL)
        self.grad_objective = self.model.generate_vector(CONTROL)
        self.q_bar = 0 

        self.comm = self.u.mpi_comm()
        self.mpi_size = self.comm.Get_size()

        self.rhs_fwd = self.model.generate_vector(STATE)
        self.rhs_adj = self.model.generate_vector(ADJOINT)
        self.rhs_adj2 = self.model.generate_vector(ADJOINT)
        self.uhat = self.model.generate_vector(STATE)
        self.phat = self.model.generate_vector(ADJOINT)
        self.zhelp = self.model.generate_vector(CONTROL)

        self.has_forward_solve = False 
        self.has_adjoint_solve = False 

    def generate_vector(self, component="ALL"):
        return self.model.generate_vector(component)

    def objective(self, z):
        self.z.zero()
        self.z.axpy(1.0, z)
        self.model.solveFwd(self.x[STATE], self.x)
        return self.model.cost(self.x)

    def computeComponents(self, z, order=0):
        """
        Computes the components for the evaluation of the cost functional

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed.
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        """

        # Check if a new control variable is used 
        # new_forward_solve = False 
        self.diff_helper.zero()
        self.diff_helper.axpy(1.0, self.z)
        self.diff_helper.axpy(-1.0, z)
        diff_norm = np.sqrt(self.diff_helper.inner(self.diff_helper))
        
        # Check if new forward solve is needed 
        if diff_norm > dl.DOLFIN_EPS or not self.has_forward_solve:
            # Update control variable (changes all samples)
            # Ask that new forward and adjoint solves are computed 
            self.z.zero()
            self.z.axpy(1.0, z)
            # new_forward_solve = True
            # logging.info("Using new forward solve")
            print("Using new forward solve")
            self.model.solveFwd(self.u, self.x) 
            
            if order >= 1: 
                logging.info("Using new adjoint solve")
                self.model.solveAdj(self.p, self.x)
                self.model.evalGradientControl(self.x, self.grad_objective)
                self.has_adjoint_solve = True
            else:
                self.has_adjoint_solve = False

        elif order >= 1 and not self.has_adjoint_solve:
            logging.info("Using new adjoint solve")
            self.model.solveAdj(self.p, self.x)
            self.model.evalGradientControl(self.x, self.grad_objective)
            self.has_adjoint_solve = True

        self.has_forward_solve = True


    def cost(self, z, order=0, **kwargs):
        """
        Computes the value of the cost functional at given control :math:`z`

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed after evaluation
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 

        :return: Value of the cost functional
        """
        self.computeComponents(z, order=order)
        objective = self.model.cost(self.x)
        if self.penalization is None:
            penalization = 0.0
        else:
            penalization = self.penalization.cost(z)
        return objective + penalization 

    def costGrad(self, g):
        """
        Computes the gradient of the cost functional 

        :param g: (Dual of) the gradient of the cost functional
        :type g: :py:class:`dolfin.Vector`

        :return: the norm of the gradient in the correct inner product :math:`(g_z,g_z)_{Z}^{1/2}`

        .. note:: Assumes :code:`self.cost` has been called with :code:`order >= 1`
        """
        self.model.evalGradientControl(self.x, g)
        if self.penalization is not None:
            self.penalization.grad(self.z, self.grad_penalization)
            g.axpy(1.0, self.grad_penalization)

        gradnorm = np.sqrt(g.inner(g))
        return gradnorm

    def costHessian(self, zhat, Hzhat):
        """
        Apply the the reduced Hessian to the vector :math:`zhat`

        :param zhat: The direction of Hessian action
        :type zhat: :py:class:`dolfin.Vector`
        :param Hzhat: The assembled Hessian action
        :type Hzhat: :py:class:`dolfin.Vector`
        
        .. note:: Assumes :code:`self.cost` has been called with :code:`order >= 2`
        """
        self.model.setLinearizationPoint(self.x)
        self.model.applyCz(zhat, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWuz(zhat, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWzz(zhat, Hzhat)

        self.model.applyCzt(self.phat, self.zhelp)
        Hzhat.axpy(1., self.zhelp)
        self.model.applyWzu(self.uhat, self.zhelp)
        Hzhat.axpy(-1., self.zhelp)

        if self.penalization is not None:
            Hzhat_penalization = self.model.generate_vector(CONTROL)
            self.penalization.hessian(self.z, zhat, Hzhat_penalization)
            Hzhat.axpy(1.0, Hzhat_penalization)




class RiskMeasureControlCostFunctional:
    """
    This class implements a risk measure cost functional for \
        optimal control problem under uncertainty

        .. math:: J(z) := \\rho[Q(m, z)] + P(z) 

    """
    def __init__(self, risk_measure, penalization=None):
        """
        Constructor

        :param risk_measure: Class implementing the risk measure :math:`\\rho(m,z)`
        :type risk_measure: :py:class:`soupy.RiskMeasure`
        :param penalization: An optional penalization object for the cost of control
        :type penalization: :py:class:`soupy.Penalization`
        """
        self.risk_measure = risk_measure
        self.penalization = penalization
        self.z = self.risk_measure.generate_vector(CONTROL)
        self.z_help = self.risk_measure.generate_vector(CONTROL)

    def generate_vector(self, component="ALL"):
        return self.risk_measure.generate_vector(component)

    def cost(self, z, order=0, **kwargs):
        """
        Computes the value of the cost functional at given control :math:`z`

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed after evaluation
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        :param kwargs: additional arguments, e.g. :code:`rng` for the risk measure computation

        :return: Value of the cost functional
        """
        self.z.zero()
        self.z.axpy(1.0, z)
        self.risk_measure.computeComponents(self.z, order=order, **kwargs)
        cost_risk = self.risk_measure.cost()
        if self.penalization is not None:
            cost_penalization = self.penalization.cost(self.z)
        else:
            cost_penalization = 0.0

        return cost_risk+cost_penalization

    def costGrad(self, g):
        """
        Computes the gradient of the cost functional 

        :param g: (Dual of) the gradient of the cost functional
        :type g: :py:class:`dolfin.Vector`

        :return: the norm of the gradient in the correct inner product :math:`(g_z,g_z)_{Z}^{1/2}`

        .. note:: Assumes :code:`self.cost` has been called with :code:`order >= 1`
        """
        g.zero()

        # Risk measure gradient
        self.risk_measure.costGrad(g)

        if self.penalization is not None:
            self.penalization.grad(self.z, self.z_help)
            g.axpy(1.0, self.z_help)

        gradnorm = np.sqrt(g.inner(g))
        return gradnorm

    def costHessian(self, zhat, Hzhat):
        """
        Apply the the reduced Hessian to the vector :math:`zhat`

        :param zhat: The direction of Hessian action
        :type zhat: :py:class:`dolfin.Vector`
        :param Hzhat: The assembled Hessian action
        :type Hzhat: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.cost` has been called with :code:`order >= 2`
        """
        Hzhat.zero()
        self.risk_measure.costHessian(zhat, Hzhat)

        if self.penalization is not None:
            self.penalization.hessian(self.z, zhat, self.z_help)
            Hzhat.axpy(1.0, self.z_help)


