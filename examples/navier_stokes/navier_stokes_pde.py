import dolfin as dl 
import ufl
import numpy as np 
import scipy.interpolate

import hippylib as hp 
import soupy 


def setup_b_splines():
    """
    Construct the expressions for the BSpline Basis 
    The degrees and total Basis are all fixed 
    """
    DEGREE = 3
    TOTAL_BASIS = 11
    NUM_BASIS_TO_USE = 10
    INITIAL_KNOTS = np.zeros(4)
    FINAL_KNOTS = np.ones(4)
    MIDDLE_KNOTS = np.linspace(0.2, 0.8, 7)
    # Create many grid points on [0, 1], 
    # i.e. [0, 0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1, 1, 1]
    # Need to repeat the boundary for times to fit the boundary spline coefficients
    knots = np.append(INITIAL_KNOTS, MIDDLE_KNOTS) 
    knots = np.append(knots, FINAL_KNOTS) 
    splines = [] 
    # Using coefficient vectors like[0, 0, ...,0, 1, 0, ..., 0] to fit cubic spline func
    # There are 9 splines to use in total. 
    for i in range(1, NUM_BASIS_TO_USE):
        coefficients = np.zeros(TOTAL_BASIS)
        coefficients[i] = 1.0
        b = scipy.interpolate.BSpline(knots, coefficients, DEGREE)
        splines.append(b)
    return splines




class VectorSplineFunction(dl.UserExpression):
    """
    Class implements a spline basis in 2D
    """

    def __init__(self, bSpline, x_start, x_end, vec, **kwargs):
        """
            - :code: `bSpline` is a scipy.interpolate bspline function
            - :code: `x_start` is the starting point for the spline (t=0)
            - :code: `x_end` is the ending point for the spline (t=1)
            - :code: `vec` is the direction for the output vector 
        """
        super().__init__(**kwargs)
        self.bSpline = bSpline 
        self.x_start = np.array(x_start)
        self.x_end = np.array(x_end)
        # define a line segnment from x_start to x_end. 
        self.d = self.x_end - self.x_start 
        # Normal vector to the line segments. 
        self.n = np.array([-self.d[1], self.d[0]])
        self.n = self.n/np.linalg.norm(self.n)
        # output vector direction, noramlized.
        self.vec = vec/np.linalg.norm(vec) # Normalized
        self.EPS = np.linalg.norm(self.d, 2) * 1e-2 # Tolerance for being "close" to the line segment.

    # Project a point x to the line segment, return the projected t value and the distance to the line segment.
    def _project_point(self, x):
        x_np = np.array([x[0], x[1]])
        t = np.inner(x_np - self.x_start, self.d)/np.inner(self.d, self.d)

        distance = np.inner(x_np - self.x_start, self.n)
        return t, np.abs(distance)
    # Evalute the point "on" the line segment using bspline basis 
    def eval(self, values, x):
        t, distance = self._project_point(x)
        if distance <= self.EPS and t >= 0 and t <= 1:
            values[0] = self.bSpline(t) * self.vec[0]
            values[1] = self.bSpline(t) * self.vec[1]
        else:
            values[0] = 0.0
            values[1] = 0.0

    def value_shape(self):
        return (2, )



class NavierStokesBluffBodyHandler:
    """
    Residual handler for the Navier stokes problem 
    """
    def __init__(self, Vh, mesh, geometry, nu, 
                 control_bases, 
                 use_stabilization=True,
                 use_nitche=True):
        self.Vh = Vh 
        self.mesh = mesh
        self.nu = nu
        self.geometry = geometry
        self.use_stabilization = use_stabilization
        self.use_nitche = use_nitche 
        self.control_bases = control_bases #B-spline basis
        self.control_bases_as_vector = dl.as_vector(self.control_bases)

        # Domain vectors
        self.x1, self.x2 = dl.SpatialCoordinate(self.mesh)
        self.e1 = dl.Constant((1.0, 0.0))
        self.e2 = dl.Constant((0.0, 1.0))

        self.Cd = dl.Constant(1e5)
        self.n = dl.FacetNormal(self.mesh)
        self.tg = ufl.perp(self.n)
        self.h = dl.CellDiameter(self.mesh)
        self.reg_norm = dl.Constant(1e-12)

    def _u_norm(self, u):
        return dl.sqrt(dl.dot(u, u) + self.reg_norm * dl.Constant(self.nu) * dl.Constant(self.nu))

    def set_viscosity(self, nu):
        self.nu = nu

    def strain(self, u):
        return dl.sym(dl.grad(u))

    def sigma_n(self, u, p, sign=1):
        return dl.dot( dl.Constant(2.0) * dl.Constant(self.nu) * self.strain(u), self.n ) - dl.Constant(sign) * p * self.n

    def tau(self, u):
        """
        Stabilization parameter
        """
        h2 = self.h**2 
        norm_u = self._u_norm(u)
        Pe =  dl.Constant(.5) * self.h * norm_u / dl.Constant(self.nu)
                    
        num = dl.Constant(1.) + dl.exp(dl.Constant(-2.)*Pe)
        den = dl.Constant(1.) - dl.exp(dl.Constant(-2.)*Pe)
        
        # [0.1 0.01]* [a1] = [ coth(.1) - 1./(.1) ]
        # [1.  0.2 ]  [a2]   [ -csch(.1)^2 + 1./(.1)^2]
        a1 = dl.Constant(0.333554921691650)
        a2 = dl.Constant(-0.004435991517475)
        tau_1 = (num/den - dl.Constant(1.)/Pe)*self.h/norm_u
        tau_2 = (a1 + a2*Pe)*dl.Constant(.5)*h2/dl.Constant(self.nu)
                
        return dl.conditional(dl.ge(Pe, .1), tau_1, tau_2)

    def all_tau(self, u):
        """
        All the 4 stabilization parameters
        """
        v, p = dl.split(u)
        norm_u = self._u_norm(u)
        h2 = self.h ** 2
        tau = [self.tau(u), h2*self._u_norm(u)]
        return tau

    def strong_residual(self, u):
        """
        The strong residual
        """
        v, p = dl.split(u)
         
        res_momentum = -dl.div(dl.Constant(2.)*dl.Constant(self.nu) * self.strain(v)) + dl.grad(v)*v + dl.grad(p)
        res_mass = dl.div( v )
        return [res_momentum, res_mass]

    def strong_linearized_residual(self, u, u_test):
        """
        The strong residual
        """
        v, p = dl.split(u)
        v_test, p_test = dl.split(u_test)
         
        res_momentum = -dl.div( dl.Constant(2.)*dl.Constant(self.nu)*self.strain(v_test) ) \
            + dl.grad(v)*v_test + dl.grad(v_test)*v
        res_mass = dl.div( v_test )
        return [res_momentum, res_mass]

    def gls_stab(self, u, u_test):
        """
        G-LS Stabilization form 
        """
        r_s = self.strong_residual(u)
        r_s_prime = self.strong_linearized_residual(u, u_test)
        tau = self.all_tau(u)
        res_stab = ( tau[0]*dl.inner(r_s[0], r_s_prime[0]) + \
                     tau[1]*dl.inner(r_s[1], r_s_prime[1]) )*dl.dx
                   
        return res_stab

    # Transform control vector to velocity on the inner boundary
    def control_to_obstacle_velocity(self, z):
        v_obstacle = dl.dot(z, self.control_bases_as_vector)
        return v_obstacle

    def parameter_to_inflow_velocity(self, m):
        v_in = dl.exp(m) * self.e1
        return v_in 

    def residual(self, u, m, u_test, z):
        v_in = self.parameter_to_inflow_velocity(m)
        v_obstacle = self.control_to_obstacle_velocity(z)

        v, p  = dl.split(u)
        v_test, p_test = dl.split(u_test)

        ns_varf = dl.Constant(2.0) * dl.Constant(self.nu) * dl.inner( self.strain(v), self.strain(v_test)) * dl.dx \
            + dl.inner(dl.grad(v) * v, v_test)*dl.dx \
            - p * dl.div(v_test) * dl.dx \
            + p_test * dl.div(v) * dl.dx

        # Only true for isotropic meshes
        hbinv = 1/self.h

        # Whether or not we use Niche's method to enforce boundary conditions. 
        # If not, then we will enforce the boundary conditions strongly, and the varf will only be the standard weak form.
        # Actually, we do use the Niche's method. 
        if self.use_nitche:
            # Dirichlet boundary around obstacle 
            nitche_obstacle = self.Cd * hbinv * dl.Constant(self.nu) * dl.inner(v - v_obstacle, v_test) * self.geometry.ds(self.geometry.OBSTACLE) \
                - dl.inner(self.sigma_n(v,p), v_test) * self.geometry.ds(self.geometry.OBSTACLE) \
                - dl.inner(self.sigma_n(v_test,p_test, sign=-1), v - v_obstacle) * self.geometry.ds(self.geometry.OBSTACLE)

            # Dirichlet Inflow boundary  
            nitche_left = self.Cd * hbinv * dl.Constant(self.nu) * dl.inner(v - v_in, v_test) * self.geometry.ds(self.geometry.LEFT) \
                - dl.inner(self.sigma_n(v,p), v_test) * self.geometry.ds(self.geometry.LEFT) \
                - dl.inner(self.sigma_n(v_test,p_test,sign=-1), v - v_in) * self.geometry.ds(self.geometry.LEFT)

            # Tangential boundary
            nitche_top = self.Cd * dl.Constant(self.nu)* hbinv * dl.dot(v, self.n)*dl.dot(v_test, self.n)*self.geometry.ds(self.geometry.TOP) \
                - dl.dot( self.sigma_n(v,p), self.n ) * dl.dot(v_test, self.n)*self.geometry.ds(self.geometry.TOP) \
                - dl.dot( self.sigma_n(v_test, p_test, sign=-1), self.n ) * dl.dot(v, self.n)*self.geometry.ds(self.geometry.TOP)

            nitche_bot = self.Cd * dl.Constant(self.nu)* hbinv * dl.dot(v, self.n)*dl.dot(v_test, self.n)*self.geometry.ds(self.geometry.BOTTOM) \
                - dl.dot( self.sigma_n(v, p), self.n ) * dl.dot(v_test, self.n)*self.geometry.ds(self.geometry.BOTTOM) \
                - dl.dot( self.sigma_n(v_test, p_test, sign=-1), self.n ) * dl.dot(v, self.n)*self.geometry.ds(self.geometry.BOTTOM)

            ns_varf += nitche_obstacle + nitche_left + nitche_top + nitche_bot 

        if self.use_stabilization: 
            # GLS Stabilization 
            gls_varf = self.gls_stab(u, u_test)
            ns_varf += gls_varf
        
        return ns_varf 

    def __call__(self, u, m, p, z):
        """
        Call varf in hippycontrol variable orderings
        """
        return self.residual(u, m, p, z)



class NavierStokesBluffBodyControlProblem(soupy.PDEVariationalControlProblem):
    """
    Rewriting of the nonlinear PDE control problem to include 
    viscosity continuation in forward solves 
    """

    def __init__(self, Vh, ns_residual, bc, bc0, 
                 max_newton_iter=10, 
                 relative_tol=1e-10,
                 verbose=False, 
                 lu_method='default'):
        # assert for class assumptions here
        assert id(Vh[hp.STATE]) == id(Vh[hp.ADJOINT]), print('Need to have same STATE and ADJOINT spaces')
        assert len(Vh) == 4
        assert Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
        self.ns_residual = ns_residual 
        self.verbose = verbose
        self.Vh = Vh
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0

        # TODO: Add tolerance 
        self.lu_method = lu_method  
        self.state_solver = soupy.NewtonBacktrackSolver()
        self.state_solver.parameters['lu_method'] = self.lu_method
        self.state_solver.parameters["print_level"] = 1 if verbose else -1

        self.A  = None
        self.At = None
        self.C = None
        self.Cz = None
        self.Wmu = None
        self.Wmm = None
        self.Wzu = None
        self.Wzz = None
        self.Wuu = None
        
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}


        # For viscosity scheduling  
        self.true_viscosity = self.ns_residual.nu
        self.viscosity_schedule = np.array([1e2, 1e1, 5, 2, 1])
        self.tolerance_schedule = np.ones(self.viscosity_schedule.shape) * 1e-10

        self.u_initial_guess_fun = dl.Function(Vh[soupy.STATE])
        self.u_test = dl.TestFunction(Vh[soupy.STATE])
        self.has_initial_guess = False
        self.n_linear_solves = 0 
        self.is_timing = False 

    def set_schedule(self, viscosity_schedule, tolerance_schedule):
        self.viscosity_schedule = viscosity_schedule
        self.tolerance_schedule = tolerance_schedule

    def varf_handler(self, u, m, p, z):
        return self.ns_residual(u, m, p, z)

    def solveFwd(self, u, x):
        """
        Forward solver using viscosity continuation. If it has a stored initial guess, then 
        the solve will try to solve the problem starting at the initial guess. If convergence fails,
        solver then proceeds with viscosity continuation 
        """
        if self.verbose:
            print(x[soupy.CONTROL].get_local())

        m_fun = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        z_fun = hp.vector2Function(x[soupy.CONTROL], self.Vh[soupy.CONTROL])
        converged = False 
        num_iters_total = 0 

        res_form = self.ns_residual(self.u_initial_guess_fun, m_fun, self.u_test, z_fun)
        jacobian_form = dl.derivative(res_form, self.u_initial_guess_fun)

        if self.has_initial_guess:
            # Solve using saved initial guess 
            num_iters, converged = self.state_solver.solve(res_form, 
                                                           self.u_initial_guess_fun, 
                                                           self.bc, 
                                                           jacobian_form)
            num_iters_total += num_iters

        if not converged:
            # Do viscosity continuation 
            self.u_initial_guess_fun.vector().zero()
            for nu_schedule, tol in zip(self.viscosity_schedule, self.tolerance_schedule):
                if self.verbose:
                    print("Continuation solve for viscosity %g" %(nu_schedule * self.true_viscosity))
                self.ns_residual.set_viscosity(nu_schedule * self.true_viscosity)
                if hasattr(self.state_solver, "set_rel_tolerance"):
                    if self.verbose:
                        print("Set relative tolerance to %g" %tol)
                    self.state_solver.set_rel_tolerance(tol)

                res_form = self.ns_residual(self.u_initial_guess_fun, m_fun, self.u_test, z_fun)
                jacobian_form = dl.derivative(res_form, self.u_initial_guess_fun)
                num_iters, converged = self.state_solver.solve(res_form, 
                                                            self.u_initial_guess_fun, 
                                                            self.bc, 
                                                            jacobian_form)

                num_iters_total += num_iters

        self.has_initial_guess = True
        if self.verbose:
            print("Setting problem back to true viscosity %g" %self.true_viscosity)
        self.ns_residual.set_viscosity(self.true_viscosity)
        self.n_linear_solves += num_iters_total
        u.zero()
        u.axpy(1.0, self.u_initial_guess_fun.vector())
    
    def set_timing_mode(self, is_timing):
        self.is_timing = True

    def setLinearizationPoint(self, x, gauss_newton):
        if self.is_timing:
            x_fun = [hp.vector2Function(x[i], self.Vh[i]) for i in range(4)]
            
            f_form = self.varf_handler(*x_fun)
            g_form = [None, None, None, None] 
            g_form[soupy.ADJOINT] = dl.derivative(f_form, x_fun[soupy.ADJOINT])
                
            self.A, dummy = dl.assemble_system(dl.derivative(g_form[soupy.ADJOINT],x_fun[soupy.STATE]), g_form[soupy.ADJOINT], self.bc0)
            self.Cz = dl.assemble(dl.derivative(g_form[soupy.ADJOINT],x_fun[soupy.CONTROL]))
            [bc.zero(self.Cz) for bc in self.bc0]
                    
            if self.solver_fwd_inc is None:
                self.solver_fwd_inc = self._createLUSolver()
            self.solver_fwd_inc.set_operator(self.A)
        else:
            super().setLinearizationPoint(x, gauss_newton)
