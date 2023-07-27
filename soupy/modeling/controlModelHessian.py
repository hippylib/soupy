from .variables import STATE, ADJOINT, PARAMETER, CONTROL 


class ControlModelHessian:
    """
    Hessian of the the QoI map with respect to the :code:`CONTROL` variable :math:`z` \
        wrapping around a :code:`soupy.ControlModel` 
    """
    def __init__(self, model):
        """
        Constructor:

        :param model: :py:class:`soupy.ControlModel`
        """
        self.model = model 
        self.uhat = self.model.generate_vector(STATE)
        self.phat = self.model.generate_vector(ADJOINT)
        self.rhs_fwd = self.model.generate_vector(STATE)
        self.rhs_adj = self.model.generate_vector(ADJOINT)
        self.rhs_adj2 = self.model.generate_vector(ADJOINT)
        self.Hz_helper = self.model.generate_vector(CONTROL)


    def mult(self, zhat, Hzhat_qoi):
        """
        Apply the Hessian of the QoI. 
        
        ::note Assumes the :code:`model.setLinearizationPoint` has been called
        """

        # Solve incremental forward
        self.model.applyCz(zhat, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)

        # Solve incremental adjoint 
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWuz(zhat, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)

        # Apply model hessian
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWzz(zhat, Hzhat_qoi)

        self.model.applyCzt(self.phat, self.Hz_helper)
        Hzhat_qoi.axpy(1., self.Hz_helper)
        self.model.applyWzu(self.uhat, self.Hz_helper)
        Hzhat_qoi.axpy(-1., self.Hz_helper)
