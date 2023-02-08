import dolfin as dl 
import numpy as np 
from ..modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL

try:
    import matplotlib.pyplot as plt 
except:
    pass

def penalizationFiniteDifference(Vh, penalization, z, dz, order=1, delta=1e-4, plotting=False):
    """
    Finite difference checks the gradient, mainly for the state variable 
    Also computes the gradients for remaining variables --- most of the time will be zero 
    """

    z1 = dl.Vector(z)
    z2 = dl.Vector(z)
    z2.axpy(delta, dz)

    p1 = penalization.cost(z1)
    p2 = penalization.cost(z2)
    fd_grad = (p2 - p1)/delta
    g1 = dl.Function(Vh[CONTROL]).vector()
    penalization.grad(z, g1)
    exact_grad = g1.inner(dz)
    print("\nFinite difference checking gradients")
    print("Analytic control derivative: %g" %exact_grad)
    print("Finite diff control derivative: %g" %fd_grad)

    if order == 2:
        print("Finite difference checking Hessians")
        Hdz = dl.Function(Vh[CONTROL]).vector()
        g2 = dl.Function(Vh[CONTROL]).vector()
        penalization.grad(z2, g2)
        penalization.hessian(z1, dz, Hdz)
        Hdz_np = Hdz.get_local()
        Hdiff_np = (g2.get_local() - g1.get_local())/delta
        print("Hdz diff norm: %g" %(np.linalg.norm(Hdz_np - Hdiff_np)))
        print("")
        
        if plotting:
            plt.figure()
            plt.plot(Hdz_np)
            plt.plot(Hdiff_np, '--')
            


