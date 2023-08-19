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
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl 

import sys, os
import hippylib as hp 

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


class ControlQoI(object):
    """
    Abstract class to define the optimization quantity of interest for the \
        optimal control problem under uncertainty \
        In the following :code:`x` will denote the variable :code:`[u, m, p, z]`, \
        denoting respectively the state :code:`u`, the parameter :code:`m`, \
        the adjoint variable :code:`p`, and the control variable :code:`z`
    
    The methods in the class ControlQoI will usually access the state :code:`u` and possibly the \
        parameter :code:`m` and control :code:`z`. The adjoint variables will never be accessed. 
    """
    
    def cost(self,x):
        """
        Given :code:`x` evaluate the cost functional. Only the state :code:`u` \
            and (possibly) the parameter :code:`m` are accessed. """
        
        raise NotImplementedError("Child class should implement method cost")
        
    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient \
            of the QoI in with respect to the state (:code:`i == soupy.STATE`), \
            parameter (:code:`i == soupy.PARAMETER`), or \
            control (:code:`i == soupy.CONTROL`).
        """

        raise NotImplementedError("Child class should implement method grad")
            
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        """
        Set the point for linearization.
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
        
    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j` = :code:`soupy.STATE`, \
            :code:`soupy.PARAMETER`, :code:`soupy.CONTROL`) \
            of the cost in direction :code:`dir`.
        """

        raise NotImplementedError("Child class should implement method apply_ij")



class L2MisfitVarfHandler:
    """
    Form handler for the :math:`L^2` Misfit 

     .. math:: \int_{\Omega} \chi (u - u_d)^2 dx

    where :math:`u_d` is the reference state \
        and :math:`\chi` is the characteristic function \
        defining the region of integration
    """

    def __init__(self, ud, chi=None):
        """
        Constructor

        :param ud: The reference state
        :type ud: :py:class:`dolfin.Function` or :py:class:`dolfin.Expression`
        :param chi: The characteristic function defining the region of integration
        :type chi: :py:class:`dolfin.Function` or :py:class:`dolfin.Expression`
        """
        self.chi = chi
        self.ud = ud

    def __call__(self, u, m, z):
        if self.chi is None:
            return (self.ud - u)**2 * dl.dx
        else:
            return self.chi*(self.ud - u)**2*dl.dx

    

class VariationalControlQoI(ControlQoI):
    """
    Class for a QoI defined by its variational form 
    """
    def __init__(self, Vh, form_handler):
        """
        Constructor

        :param Vh: List of function spaces for the state, parameter, 
            adjoint, and optimization variables
        :type Vh: list of :py:class:`dolfin.FunctionSpace`
        :param form_handler: The form handler for the variational form with a 
            :code:`__call__` method that takes as input the state, parameter, and control variables
            as functions and returns the variational form 
        """
        self.Vh = Vh
        self.x = [dl.Function(Vh[STATE]).vector(), dl.Function(Vh[PARAMETER]).vector(),
                  dl.Function(Vh[ADJOINT]).vector(), dl.Function(Vh[CONTROL]).vector()]
        self.x_test = [dl.TestFunction(Vh[STATE]), dl.TestFunction(Vh[PARAMETER]),
                       dl.TestFunction(Vh[ADJOINT]), dl.TestFunction(Vh[CONTROL])]

        self.form_handler = form_handler

    def cost(self, x):
        """
        Evaluate the qoi at given point :math:`q(u,m,z)`

        :param x: List of vectors :code:`[u, m, p, z]` representing the state, 
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :return: QoI evaluated at x
        """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])

        return dl.assemble(self.form_handler(u, m, z))

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian functional \
            with respect to the state u).

        :param x: List of vectors :code:`[u, m, p, z]` representing the state, 
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :param rhs: The assembled rhs for the adjoint problem.
        :type rhs: :py:class:`dolfin.Vector`
        """
        self.grad(STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
        """
        First variation of the QoI with respect to the :code:`i` th variable \
            where :code:`i` is either :code:`soupy.STATE`, :code:`soupy.PARAMETER`, \
            or :code:`soupy.CONTROL`.

        :param i: Index of the variable with respect to which the first variation is taken
        :type i: int
        :param x: List of vectors :code:`[u, m, p, z]` representing the state,
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :param out: The assembled first variation 
        :type out: :py:class:`dolfin.Vector`
        """
        out.zero()
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        x_fun = [u, m, None, z]
        f_form = self.form_handler(u, m, z)
        f = dl.assemble(dl.derivative(f_form, x_fun[i], self.x_test[i]))
        out.axpy(1.0, f)


    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\\delta_{ij}` (:code:`i,j` = :code:`soupy.STATE`, \
            :code:`soupy.PARAMETER`, :code:`soupy.CONTROL`) \
            of the QoI in direction :code:`dir`.

        :param i: Index of the output variable
        :type i: int
        :param j: Index of the input variable
        :type j: int
        :param dir: The direction in which to apply the second variation
        :type dir: :py:class:`dolfin.Vector`
        :param out: The assembled second variation
        :type out: :py:class:`dolfin.Vector`

        ..note:: :code:`setLinearizationPoint` must be called before calling this method.
        """

        out.zero()

        x_fun = [hp.vector2Function(self.x[s], self.Vh[s]) for s in range(len(self.x))]
        f_form = self.form_handler(x_fun[STATE], x_fun[PARAMETER], x_fun[CONTROL])
        dir_fun = hp.vector2Function(dir, self.Vh[j])

        f_i = dl.derivative(f_form, x_fun[i], self.x_test[i])
        f_ij = dl.derivative(f_i, x_fun[j], dir_fun)
        out.axpy(1.0, dl.assemble(f_ij))


    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the QoI in the \
            :code:`i` th, :code:`j` th, and :code:`k` th variables in directions \
            :code:`dir1` and :code:`dir2`.

        :param i: First variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type i: int
        :param j: Second variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type j: int
        :param k: Third variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type k: int
        :param dir1: Direction for variable :code:`j`
        :type dir1: :py:class:`dolfin.Vector`
        :param dir2: Direction for variable :code:`k`
        :type dir2: :py:class:`dolfin.Vector`
        :param out: The assembled third variation
        :type out: :py:class:`dolfin.Vector`
        """
        out.zero()

        x_fun = [hp.vector2Function(self.x[s], self.Vh[s]) for s in range(len(self.x))]
        f_form = self.form_handler(x_fun[STATE], x_fun[PARAMETER], x_fun[CONTROL])
        dir1_fun, dir2_fun = hp.vector2Function(dir1, self.Vh[i]), hp.vector2Function(dir2, self.Vh[j])

        f_i = dl.derivative(f_form, x_fun[i], dir1_fun)
        f_ij = dl.derivative(f_i, x_fun[j], dir2_fun)
        f_ijk = dl.derivative(f_ij, x_fun[k], self.x_test[k])
        out.axpy(1.0, dl.assemble(f_ijk))

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        :param x: List of vectors :code:`[u, m, p, z]` representing the state,
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])


class L2MisfitControlQoI(ControlQoI):
    """
    Class for the :math:`L^2(\Omega)` misfit functional,
    
    .. math:: \\int_\\Omega (u - u_d)^2 dx

    where :math:`u_d` is the reference state.
    """
    def __init__(self, Vh, ud):
        """
        Constructor

        :param Vh: List of function spaces for the state, parameter, 
            adjoint, and optimization variables
        :type Vh: list of :py:class:`dolfin.FunctionSpace`
        :param ud: The reference state as a vector
        :type ud: :py:class:`dolfin.Vector`
        """
        self.Vh = Vh
        self.x = [dl.Function(Vh[STATE]).vector(), dl.Function(Vh[PARAMETER]).vector(),
                  dl.Function(Vh[ADJOINT]).vector(), dl.Function(Vh[CONTROL]).vector()]
        self.x_test = [dl.TestFunction(Vh[STATE]), dl.TestFunction(Vh[PARAMETER]),
                       dl.TestFunction(Vh[ADJOINT]), dl.TestFunction(Vh[CONTROL])]

        self.ud = ud
        self.diff= dl.Function(Vh[STATE]).vector()
        self.Mdiff = dl.Function(Vh[STATE]).vector()

        u_trial = dl.TrialFunction(Vh[STATE])
        u_test = dl.TestFunction(Vh[STATE])
        self.M_STATE = dl.assemble(dl.inner(u_trial, u_test)*dl.dx)

    def cost(self, x):
        """
        Evaluate the qoi at given point :math:`q(u,m,z)`

        :param x: List of vectors :code:`[u, m, p, z]` representing the state, 
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :return: QoI evaluated at x
        """

        self.diff.zero()
        self.diff.axpy(1.0, x[STATE])
        self.diff.axpy(-1.0, self.ud)
        self.M_STATE.mult(self.diff, self.Mdiff)
        return self.diff.inner(self.Mdiff)

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian functional \
            with respect to the state u).

        :param x: List of vectors :code:`[u, m, p, z]` representing the state, 
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :param rhs: The assembled rhs for the adjoint problem.
        :type rhs: :py:class:`dolfin.Vector`
        """
        self.grad(STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
        """
        First variation of the QoI with respect to the :code:`i` th variable \
            where :code:`i` is either :code:`soupy.STATE`, :code:`soupy.PARAMETER`, \
            or :code:`soupy.CONTROL`.

        :param i: Index of the variable with respect to which the first variation is taken
        :type i: int
        :param x: List of vectors :code:`[u, m, p, z]` representing the state,
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        :param out: The assembled first variation 
        :type out: :py:class:`dolfin.Vector`
        """

        out.zero()
        if i == STATE:
            self.diff.zero()
            self.diff.axpy(1.0, x[STATE])
            self.diff.axpy(-1.0, self.ud)
            self.M_STATE.mult(self.diff, self.Mdiff)
            out.axpy(2.0, self.Mdiff)



    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\\delta_{ij}` (:code:`i,j` = :code:`soupy.STATE`, \
            :code:`soupy.PARAMETER`, :code:`soupy.CONTROL`) \
            of the QoI in direction :code:`dir`.

        :param i: Index of the output variable
        :type i: int
        :param j: Index of the input variable
        :type j: int
        :param dir: The direction in which to apply the second variation
        :type dir: :py:class:`dolfin.Vector`
        :param out: The assembled second variation
        :type out: :py:class:`dolfin.Vector`

        ..note:: :code:`setLinearizationPoint` must be called before calling this method.
        """

        out.zero()

        if i == STATE and j == STATE:
            self.M_STATE.mult(dir, self.Mdiff)
            out.axpy(2.0, self.Mdiff)



    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the QoI in the \
            :code:`i` th, :code:`j` th, and :code:`k` th variables in directions \
            :code:`dir1` and :code:`dir2`.

        :param i: First variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type i: int
        :param j: Second variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type j: int
        :param k: Third variable index (:code:`soupy.STATE`, :code:`soupy.PARAMETER`, or :code:`soupy.CONTROL`)
        :type k: int
        :param dir1: Direction for variable :code:`j`
        :type dir1: :py:class:`dolfin.Vector`
        :param dir2: Direction for variable :code:`k`
        :type dir2: :py:class:`dolfin.Vector`
        :param out: The assembled third variation
        :type out: :py:class:`dolfin.Vector`
        """
        out.zero()


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        :param x: List of vectors :code:`[u, m, p, z]` representing the state,
            parameter, adjoint, and control variables
        :type x: list of :py:class:`dolfin.Vector`
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])

