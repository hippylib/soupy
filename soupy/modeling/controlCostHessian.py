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

        ..math:: z_21^T H z_2
        """
        self.mult(z1, self.z_help)
        return self.z_help.inner(z2)



    def mult(self, zhat, out):
        """
        Apply the Hessian of the QoI. 
        
        ::note Assumes the :code:`model.setLinearizationPoint` has been called
        """
        self.cost_functional.costHessian(zhat, out)
        self._ncalls += 1 



