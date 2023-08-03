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


class ControlModelHessian:
    """
    Hessian of the the QoI map with respect to the :code:`CONTROL` variable :math:`z` \
        wrapping around a :code:`soupy.ControlModel` 
    """
    def __init__(self, model):
        """
        Constructor:

        :param model: Control model defining the control to QoI map
        :type model: :py:class:`soupy.ControlModel`
        """
        self.model = model 
        self.uhat = self.model.generate_vector(STATE)
        self.phat = self.model.generate_vector(ADJOINT)
        self.rhs_fwd = self.model.generate_vector(STATE)
        self.rhs_adj = self.model.generate_vector(ADJOINT)
        self.rhs_adj2 = self.model.generate_vector(ADJOINT)
        self.Hz_helper = self.model.generate_vector(CONTROL)

    def init_vector(self, z, dim):
        """
        Initialize vector to be compatible with operator. Note since \
            the Hessian is symmetric, dimension is not important 
        """
        self.model.init_control(z)


    def mult(self, zhat, Hzhat):
        """
        Apply the Hessian of the QoI. 

        :param zhat: Direction for Hessian action 
        :type zhat: :py:class:`dolfin.Vector` or similar
        :param Hzhat: Output for Hessian action
        :type Hzhat: :py:class:`dolfin.Vector` or similar
        
        ::note Assumes the :code:`model.setLinearizationPoint` has been called
        """
        Hzhat.zero()

        # Solve incremental forward
        self.model.applyCz(zhat, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)

        # Solve incremental adjoint 
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWuz(zhat, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)

        # Apply model hessian
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWzz(zhat, Hzhat)

        self.model.applyCzt(self.phat, self.Hz_helper)
        Hzhat.axpy(1., self.Hz_helper)
        self.model.applyWzu(self.uhat, self.Hz_helper)
        Hzhat.axpy(-1., self.Hz_helper)
