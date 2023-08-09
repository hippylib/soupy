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


from .variables import STATE, ADJOINT, PARAMETER, CONTROL 

class ControlCostHessian:
    """
    Hessian of the the QoI map with respect to the :code:`CONTROL` variable :math:`z` \
        wrapping around a :code:`soupy.ControlModel`. Hessian will be applied where \
        :code:`soupy.ControlModel` is last called with :code:`order>=2`
    """
    def __init__(self, cost_functional):
        """
        Constructor:

        :param cost_functional: cost functional
        :type cost_functional: :py:class:`soupy.ControlCostFunctional`
        """
        self.cost_functional = cost_functional
        self.z_help = self.cost_functional.generate_vector(CONTROL)
        self._ncalls = 0 

    @property
    def ncalls(self):
        return self._ncalls 

    def inner(self, z1, z2):
        """
        Computes the Hessian weighted inner product 

        ..math:: z_1^T H z_2

        :param z1: Vector 1
        :type z1: :py:class:`dolfin.Vector` or similar
        :param z2: Vector 2
        :type z2: :py:class:`dolfin.Vector` or similar

        :returns: The Hessian weighted inner product
        """
        self.mult(z1, self.z_help)
        return self.z_help.inner(z2)



    def mult(self, zhat, Hzhat):
        """
        Apply the Hessian of the QoI in direction :math:`\hat{z}`

        :param zhat: Direction for Hessian action 
        :type zhat: :py:class:`dolfin.Vector` or similar
        :param Hzhat: Output for Hessian action
        :type Hzhat: :py:class:`dolfin.Vector` or similar
        """

        self.cost_functional.hessian(zhat, Hzhat)
        self._ncalls += 1 



