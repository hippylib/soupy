
import dolfin as dl
import math
import numpy as np

import sys

sys.path.append('../../')
from hippycontrol.modeling.variables import STATE, PARAMETER, ADJOINT, CONTROL

def poisson_control_settings():
    settings = {}
    settings['nx'] = 32
    settings['ny'] = 32

    settings['STRENGTH_UPPER'] = 10.
    settings['STRENGTH_LOWER'] = -10.
    settings['LINEAR'] = True


    settings['N_WELLS_PER_SIDE'] = 5
    settings['LOC_LOWER'] = 0.25
    settings['LOC_UPPER'] = 0.75
    settings['WELL_WIDTH'] = 0.1

    settings['GAMMA'] = 0.1
    settings['DELTA'] = 0.5
    settings['THETA0'] = 2.0
    settings['THETA1'] = 0.5
    settings['ALPHA'] = math.pi/4

    return settings


class PoissonVarfHandler:
    """
    """
    def __init__(self,Vh,settings = poisson_control_settings()):
        """
        """
        self.linear = settings['LINEAR']

        # Set up the control right hand side
        well_grid = np.linspace(settings['LOC_LOWER'],settings['LOC_UPPER'],settings['N_WELLS_PER_SIDE'])
        well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
        mollifier_list = [] 

        for i in range(settings['N_WELLS_PER_SIDE']):
            for j in range(settings['N_WELLS_PER_SIDE']):
                mollifier_list.append(
                        dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
                            xi=well_grid[i], 
                            yj=well_grid[j], 
                            a=1/(2*math.pi*settings['WELL_WIDTH']**2),
                            b=settings['WELL_WIDTH'],
                            degree=2)
                        )

        self.mollifiers = dl.as_vector(mollifier_list)

        # The only reason I am passing Vh in now is in order to do an assertion sanity check
        # on the control variable. Vh[CONTROL] needs to be accessed outside of the varf_handler
        # because it needs to be passed into the PDEVariationalControlProblem constructor
        # so it doesn't make sense to me to define Vh[CONTROL] here..

        assert Vh[CONTROL].dim() == len(mollifier_list)


    def __call__(self,u,m,p,z):
        if self.linear:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx - dl.inner(self.mollifiers,z)*p*dl.dx
        else:
            return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx
    

class NonlinearPoissonVarfHandler:
    """
    """
    def __init__(self,Vh,settings = poisson_control_settings()):
        """
        """
        self.linear = settings['LINEAR']

        # Set up the control right hand side
        well_grid = np.linspace(settings['LOC_LOWER'],settings['LOC_UPPER'],settings['N_WELLS_PER_SIDE'])
        well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
        mollifier_list = [] 

        for i in range(settings['N_WELLS_PER_SIDE']):
            for j in range(settings['N_WELLS_PER_SIDE']):
                mollifier_list.append(
                        dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
                            xi=well_grid[i], 
                            yj=well_grid[j], 
                            a=1/(2*math.pi*settings['WELL_WIDTH']**2),
                            b=settings['WELL_WIDTH'],
                            degree=2)
                        )

        self.mollifiers = dl.as_vector(mollifier_list)

        # The only reason I am passing Vh in now is in order to do an assertion sanity check
        # on the control variable. Vh[CONTROL] needs to be accessed outside of the varf_handler
        # because it needs to be passed into the PDEVariationalControlProblem constructor
        # so it doesn't make sense to me to define Vh[CONTROL] here..

        assert Vh[CONTROL].dim() == len(mollifier_list)


    def __call__(self,u,p,m,z):
        return dl.exp(m)*dl.inner(dl.grad(u),dl.grad(p))*dl.dx + u**3*p*dl.dx  - dl.inner(self.mollifiers,z)*p*dl.dx

    def residual(self, u, u_test, m, z):
        return self.__call__(u, u_test, m, z)

    def Jacobian(self, u, u_test, u_trial, m, e):
        """
        Returns the weak form of the Jacobian matrix.
        
        INPUTS:
        - x: the state at which evaluate the Jacobian
        - x_test: the test function
        - x_trial: the trial function
        - m: the uncertain parameter
        """
        r = self.residual(u, u_test, m, e)
        return dl.derivative(r,u,u_trial)


    def mass(self, u, u_test):
        """
        returns the mass matrix 
        """
        return u * u_test * dl.dx 

    def stiffness(self, u, u_test):
        """
        returns the stiffness matrix 
        """
        return dl.inner(dl.grad(u), dl.grad(u_test)) * dl.dx 
