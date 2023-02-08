import dolfin as dl 

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


class ControlQoI(object):
    """
    Abstract class to model the control quantity of a control problem
    In the following :code:`x` will denote the variable :code:`[u, m, p, z]`, denoting respectively 
    the state :code:`u`, the parameter :code:`m`, the adjoint variable :code:`p`, and the control variable :code:`z`
    
    The methods in the class ControlQoI will usually access the state u and possibly the
    parameter :code:`m` and control :code: `z`. The adjoint variables will never be accessed. 
    """
    
    def cost(self,x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        
        raise NotImplementedError("Child class should implement method cost")
        
    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """

        raise NotImplementedError("Child class should implement method grad")
            
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        """
        Set the point for linearization.
        Inputs:
        
            :code:`x=[u, m, p]` - linearization point
            :code:`gauss_newton_approx (bool)` - whether to use Gauss Newton approximation 
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
        
    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """

        raise NotImplementedError("Child class should implement method apply_ij")



class L2MisfitVarfHandler:
    """
    Form handler for the L2 Misfit 
    """

    def __init__(self, ud, chi=None):
        """
        :code: `ud` is the reference function 
        :code: `chi` is a characteristic function defining region of integration
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
    define the quantity of interest and its derivative information
    """
    def __init__(self, mesh, Vh, form_handler):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for [state, parameter, adjoint, optimization] variable
        """
        self.mesh = mesh
        self.Vh = Vh
        self.x = [dl.Function(Vh[STATE]).vector(), dl.Function(Vh[PARAMETER]).vector(),
                  dl.Function(Vh[ADJOINT]).vector(), dl.Function(Vh[CONTROL]).vector()]
        self.x_test = [dl.TestFunction(Vh[STATE]), dl.TestFunction(Vh[PARAMETER]),
                       dl.TestFunction(Vh[ADJOINT]), dl.TestFunction(Vh[CONTROL])]

        self.form_handler = form_handler

    def cost(self, x):
        """
        evaluate the qoi at given x
        :param x: [state, parameter, adjoint, optimization] variable
        :return: qoi(x)
        """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])

        return dl.assemble(self.form_handler(u, m, z))

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).
        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad(STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
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
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,CONTROL) of the q.o.i. in direction dir.
        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, CONTROL=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.
        NOTE: setLinearizationPoint must be called before calling this method.
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
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or CONTROL
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
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
        INPUTS:
        - x = [u,m,p,z] is a list of the state u, parameter m, and adjoint variable p
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])


class L2MisfitControlQoI(ControlQoI):
    """
    define the quantity of interest and its derivative information
    """
    def __init__(self, mesh, Vh, ud):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for [state, parameter, adjoint, optimization] variable
        """
        self.mesh = mesh
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
        evaluate the qoi at given x
        :param x: [state, parameter, adjoint, optimization] variable
        :return: qoi(x)
        """
        self.diff.zero()
        self.diff.axpy(1.0, x[STATE])
        self.diff.axpy(-1.0, self.ud)
        self.M_STATE.mult(self.diff, self.Mdiff)
        return self.diff.inner(self.Mdiff)

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).
        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad(STATE, x, rhs)
        rhs *= -1

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            self.diff.zero()
            self.diff.axpy(1.0, x[STATE])
            self.diff.axpy(-1.0, self.ud)
            self.M_STATE.mult(self.diff, self.Mdiff)
            out.axpy(2.0, self.Mdiff)



    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,CONTROL) of the q.o.i. in direction dir.
        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, CONTROL=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.
        NOTE: setLinearizationPoint must be called before calling this method.
        """

        out.zero()

        if i == STATE and j == STATE:
            self.M_STATE.mult(dir, self.Mdiff)
            out.axpy(2.0, self.Mdiff)



    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or CONTROL
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
        """
        out.zero()


    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.
        INPUTS:
        - x = [u,m,p,z] is a list of the state u, parameter m, and adjoint variable p
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])
