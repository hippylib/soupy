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

class RiskMeasure:

    """
    Abstract class for the risk measure :math:`\\rho[Q](z)` 
    """
    def generate_vector(self, components="ALL"):
        """
        If :code:`components` is :code:`STATE`, :code:`PARAMETER`, :code:`ADJOINT`, \
            or :code:`CONTROL`, return a vector corresponding to that function space. \
            If :code:`components` is :code:`"ALL"`, \
            Generate the list of vectors :code:`x = [u,m,p,z]`
        """
        raise NotImplementedError("Child class should implement method generate_vector")

    def computeComponents(self, z, order=0):
        """
        Computes the components for the evaluation of the risk measure

        :param z: the control variable
        :type z: :py:class:`dolfin.Vector`
        :param order: Order of the derivatives needed.
            :code:`0` for cost, :code:`1` for gradient, :code:`2` for Hessian
        :type order: int 
        """
        raise NotImplementedError("Child class should implement method computeComponents")

    def cost(self):
        """
        Evaluates the value of the risk measure once components have been computed

        :return: Value of the cost functional

        .. note:: Assumes :code:`computeComponents` has been called with :code:`order>=0`
        """
        raise NotImplementedError("Child class should implement method costValue")

    def grad(self):
        """
        Evaluates the gradient of the risk measure once components have been computed

        :param g: (Dual of) the gradient of the risk measure to store result in
        :type g: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        raise NotImplementedError("Child class should implement method grad")

    def hessian(self, zhat, Hzhat):
        """
        Apply the hessian of the risk measure once components have been computed \
            in the direction :code:`zhat`
        
        :param zhat: Direction for application of Hessian action of the risk measure
        :type zhat: :py:class:`dolfin.Vector`
        :param Hzhat: (Dual of) Result of the Hessian action of the risk measure 
            to store the result in
        :type Hzhat: :py:class:`dolfin.Vector`

        .. note:: Assumes :code:`self.computeComponents` has been called with :code:`order >= 1`
        """
        raise NotImplementedError("Child class should implement method costHess")



