import dolfin as dl 
import numpy as np 
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 

from ..modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL
from ..modeling.augmentedVector import AugmentedVector

def stochasticCostFiniteDifference(pde_cost, z, dz, delta=1e-3, sample_size=1):
    gz = pde_cost.generate_vector(CONTROL)

    if isinstance(z, AugmentedVector):
        z0 = AugmentedVector(z.get_vector()) 
        z0.set_scalar(z.get_scalar())

        z1 = AugmentedVector(z.get_vector()) 
        z1.set_scalar(z.get_scalar())
        
    else:
        z0 = dl.Vector(z)
        z1 = dl.Vector(z)
    
    rng = hp.Random()
    c0 = pde_cost.cost(z0, rng=rng, order=1, sample_size=sample_size)
    pde_cost.costGrad(z0, gz)

    rng = hp.Random()
    delta = 1e-3
    z1.axpy(delta, dz)
    c1 = pde_cost.cost(z1, rng=rng, order=0, sample_size=sample_size)

    fd_grad = (c1 - c0)/delta
    ad_grad = gz.inner(dz)

    print("Analytic gradient: %g" %ad_grad)
    print("Finite diff gradient: %g" %fd_grad)

def SAACostFiniteDifference(pde_cost, z, dz, delta=1e-3):
    gz = pde_cost.generate_vector(CONTROL)

    if isinstance(z, AugmentedVector):
        z0 = AugmentedVector(z.get_vector()) 
        z0.set_scalar(z.get_scalar())

        z1 = AugmentedVector(z.get_vector()) 
        z1.set_scalar(z.get_scalar())
        
    else:
        z0 = dl.Vector(z)
        z1 = dl.Vector(z)
    
    rng = hp.Random()
    c0 = pde_cost.cost(z0, order=1)
    pde_cost.costGrad(z0, gz)

    rng = hp.Random()
    delta = 1e-3
    z1.axpy(delta, dz)
    c1 = pde_cost.cost(z1, order=0)

    fd_grad = (c1 - c0)/delta
    ad_grad = gz.inner(dz)

    print("Analytic gradient: %g" %ad_grad)
    print("Finite diff gradient: %g" %fd_grad)