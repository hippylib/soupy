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

import numpy as np 
import dolfin as dl

import sys, os
import hippylib as hp

import ufl

from .variables import STATE, PARAMETER, ADJOINT, CONTROL


class PDEVariationalControlProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False, lu_method="mumps"):
        """
        Constructor 

        :param Vh: List of function spaces the state, parameter, adjoint, and control
        :type Vh: list of :py:class:`dolfin.FunctionSpace`
        :param varf_handler: Variational form handler with :code:`__call__` method
        :param bc: List of Dirichlet boundary conditions for the state
        :param bc0: List of zeroed Dirichlet boundary conditions 
        :param is_fwd_linear: Flag indicating whether the forward problem is linear
        :type is_fwd_linear: bool
        :param lu_method: Method for solving linear systems (default, mumps, etc.)
        :type lu_method: str
        """

        # assert for class assumptions here
        assert id(Vh[STATE]) == id(Vh[ADJOINT]), print('Need to have same STATE and ADJOINT spaces')
        assert len(Vh) == 4
        # assert Vh[STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')

        self.Vh = Vh
        self.varf_handler = varf_handler
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
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
        self.lu_method = lu_method
        
        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}
        self.n_linear_solves = 0 
        self.nonlinear_solver_parameters = None 
    
    def set_nonlinear_solver_parameters(self, parameters):
        """ Set the solver parameters used for `dolfin.NonlinearVariationalSolver`

        :param parameters: Solver parameters for `dolfin.NonlinearVariationalSolver`
        :type parameters: dict
        """
        self.nonlinear_solver_parameters = parameters

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        return dl.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return dl.Function(self.Vh[PARAMETER]).vector()

    def generate_control(self):
        return dl.Function(self.Vh[CONTROL]).vector()
    
    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        # This should be addressed at some point
        # m.init( dummy.mpi_comm(), dummy.local_range() )
        m.init( dummy.local_range() )

    def init_control(self, z):
        """ Initialize the parameter. """
        dummy = self.generate_control()
        # This should be addressed at some point
        # z.init( dummy.mpi_comm(), dummy.local_range() )
        z.init(dummy.local_range())
    
    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m, z`, find :math:`u` such that
        
            .. math:: \\delta_p F(u, m, p, z;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
            res_form = self.varf_handler(u, m, p, z)
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
            self.solver.set_operator(A)
            self.solver.solve(state, b)
            self.n_linear_solves += 1 
        else:
            u = hp.vector2Function(x[STATE], self.Vh[STATE])
            m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
            res_form = self.varf_handler(u, m, p, z)
            jacobian_form = dl.derivative(res_form, u) 
            nonlinear_problem = dl.NonlinearVariationalProblem(res_form, u, self.bc, jacobian_form)
            solver = dl.NonlinearVariationalSolver(nonlinear_problem)

            if self.nonlinear_solver_parameters is not None:
                solver.parameters.update(self.nonlinear_solver_parameters)

            num_iters, converged = solver.solve()
            state.zero()
            state.axpy(1., u.vector())
            self.n_linear_solves += num_iters
        

    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m, z, u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p, z;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
            
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p, z)
        adj_form = dl.derivative( dl.derivative(varf, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, ufl.inner(u,du)*ufl.dx, self.bc0)
        self.solver.set_operator(Aadj)

        # Apply the zeroed Dirichlet boundary conditions 
        # This is safer then the standard hippylib approach where 
        # the boundary conditions are assumed to have been applied
        # before being passed into the :code:`solveAdj` method
        local_adj_rhs = adj_rhs.copy()
        for bc0 in self.bc0:
            bc0.apply(local_adj_rhs)

        self.solver.solve(adj, local_adj_rhs)
        self.n_linear_solves += 1 
     
    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p, z`; evaluate :math:`\\delta_m F(u, m, p, z; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)

    def evalGradientControl(self, x, out):
        """Given :math:`u, m, p, z`; evaluate :math:`\\delta_z F(u, m, p, z; \\hat{z}),\\, \\forall \\hat{z}.` """
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        dz = dl.TestFunction(self.Vh[CONTROL])
        res_form = self.varf_handler(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, z, dz), tensor=out)
         
    # The following function computes the Jacobian of the PDE residual at the point x and set up the incremental forward and adjoint solvers.
    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        # Here x = [u, m, p, z] is the point at which we want to linearize the PDE and compute the Hessian action.
        x_fun = [hp.vector2Function(x[i], self.Vh[i]) for i in range(4)]
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None,None,None,None]
        for i in range(4):
            g_form[i] = dl.derivative(f_form, x_fun[i])
            
        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0) # The second g_form[ADJOINT] is just a place holder for RHS
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        self.Cz = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[CONTROL]))
        [bc.zero(self.C) for bc in self.bc0]
        [bc.zero(self.Cz) for bc in self.bc0]

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
            self.Wzu = None
            self.Wzz = None
        else:
            self.Wuu = self._assemble_matrix_reuse(
                dl.derivative(g_form[STATE],x_fun[STATE]),
                self.Wuu,
            )
            # bc.zero can be used the set the rows corresponding to the d.o.f. of solution's Dirichlet B.C. to zero 
            # Since we have used full space for all function spaces, we have to set the coresponding rows and columns to zero. d
            [bc.zero(self.Wuu) for bc in self.bc0]
            self._zero_columns_in_place(self.Wuu)
            self.Wmu = self._assemble_matrix_reuse(
                dl.derivative(g_form[PARAMETER],x_fun[STATE]),
                self.Wmu,
            )
            self._zero_columns_in_place(self.Wmu)
            self.Wmm = self._assemble_matrix_reuse(
                dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]),
                self.Wmm,
            )

            self.Wzu = self._assemble_matrix_reuse(
                dl.derivative(g_form[CONTROL],x_fun[STATE]),
                self.Wzu,
            )
            self._zero_columns_in_place(self.Wzu)
            self.Wzz = self._assemble_matrix_reuse(
                dl.derivative(g_form[CONTROL],x_fun[CONTROL]),
                self.Wzz,
            )



            # self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            # # bc.zero can be used the set the rows corresponding to the d.o.f. of solution's Dirichlet B.C. to zero 
            # # Since we have used full space for all function spaces, we have to set the coresponding rows and columns to zero. d
            # [bc.zero(self.Wuu) for bc in self.bc0]
            # # print("WUU NORM", np.linalg.norm(self.Wuu.array()))
            # Wuu_t = hp.Transpose(self.Wuu)
            # [bc.zero(Wuu_t) for bc in self.bc0]
            # self.Wuu = hp.Transpose(Wuu_t)
            # self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            # Wmu_t = hp.Transpose(self.Wmu)
            # [bc.zero(Wmu_t) for bc in self.bc0]
            # self.Wmu = hp.Transpose(Wmu_t)
            # self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]))

            # self.Wzu = dl.assemble(dl.derivative(g_form[CONTROL],x_fun[STATE]))
            # Wzu_t = hp.Transpose(self.Wzu)
            # [bc.zero(Wzu_t) for bc in self.bc0]
            # self.Wzu = hp.Transpose(Wzu_t)
            # self.Wzz = dl.assemble(dl.derivative(g_form[CONTROL],x_fun[CONTROL]))
        
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj == False`:

            Solve the forward incremental system:
            Given :math:`u, m, z`, find :math:`\\tilde{u}` such that
            
                .. math:: \\delta_{pu} F(u, m, p, z ; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.
            
            If :code:`is_adj == True`:

            Solve the adjoint incremental system:
            Given :math:`u, m, z`, find :math:`\\tilde{p}` such that
            
                .. math:: \\delta_{up} F(u, m, p, z; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """

        # Apply the zeroed Dirichlet boundary conditions 
        # This is safer then the standard hippylib approach where 
        # the boundary conditions are assumed to have been applied
        # before being passed into the :code:`solveIncremental` method
        local_rhs = rhs.copy()
        for bc0 in self.bc0:
            bc0.apply(local_rhs)

        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(out, local_rhs)
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(out, local_rhs)
        self.n_linear_solves += 1 
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p, z`; compute 
            :math:`\\delta_{ij} F(u, m, p, z; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C

        KKT[CONTROL, STATE] = self.Wzu
        KKT[CONTROL, CONTROL] = self.Wzz

        if i == ADJOINT and j == CONTROL and self.Cz is not None:
            # Check Cz first since the index ordering is different with CONTROL 
            # This avoids constructing an explicit transpose of Cz.
            self.Cz.mult(dir, out) 
        elif i == CONTROL and j == ADJOINT and self.Cz is not None:
            self.Cz.transpmult(dir, out)

        elif i >= j:
            if KKT[i,j] is None:
                out.zero()
            else:
                KKT[i,j].mult(dir, out)
        else:
            if KKT[j,i] is None:
                out.zero()
            else:
                KKT[j,i].transpmult(dir, out)
                
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        x_fun = [hp.vector2Function(x[ii], self.Vh[ii]) for ii in range(4)]
        idir_fun = dl.TestFunction(self.Vh[i])
        jdir_fun = hp.vector2Function(jdir, self.Vh[j])
        kdir_fun = hp.vector2Function(kdir, self.Vh[k])
        
        res_form = self.varf_handler(*x_fun)
        form = dl.derivative(
               dl.derivative(
               dl.derivative(res_form, x_fun[i], idir_fun),
               x_fun[j], jdir_fun),
               x_fun[k], kdir_fun)
        
        out.zero()
        dl.assemble(form, tensor=out)
        
        if i in [STATE,ADJOINT]:
            [bc.apply(out) for bc in self.bc0]
                   
    def _createLUSolver(self):   
        if hasattr(self, 'lu_method'):
            return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm(), method=self.lu_method)
        else:
            return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm(), method="default")

    def _assemble_matrix_reuse(self, form, tensor):
        if tensor is None:
            return dl.assemble(form)

        dl.assemble(form, tensor=tensor)
        return tensor

    def _zero_columns_in_place(self, matrix):
        col_scale = dl.Vector(self.Vh[STATE].mesh().mpi_comm())
        matrix.init_vector(col_scale, 1)

        local_range = col_scale.local_range()
        local_values = np.ones(local_range[1] - local_range[0], dtype=np.float64)

        constrained = set()
        for bc in self.bc0:
            constrained.update(bc.get_boundary_values().keys())

        for dof in constrained:
            if local_range[0] <= dof < local_range[1]:
                local_values[dof - local_range[0]] = 0.0

        col_scale.set_local(local_values)
        col_scale.apply("")

        matrix_backend = dl.as_backend_type(matrix)
        matrix_mat = matrix_backend.mat() if hasattr(matrix_backend, "mat") else matrix_backend
        scale_backend = dl.as_backend_type(col_scale)
        scale_vec = scale_backend.vec() if hasattr(scale_backend, "vec") else scale_backend
        matrix_mat.diagonalScale(None, scale_vec)
        matrix_mat.assemble()

    # ------------------------------------------------------------------
    # Helpers for higher-order Taylor approximations

    def forSolveAdjIncrementalAdj(self, x, mhat):
        u_fun = hp.vector2Function(x[STATE], self.Vh[STATE])
        m_fun = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p_fun = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        mhat_fun = hp.vector2Function(mhat, self.Vh[PARAMETER])

        form = self.varf_handler(u_fun, m_fun, p_fun, z_fun)
        p_test = dl.TestFunction(self.Vh[ADJOINT])

        dmr = dl.derivative(form, m_fun, mhat_fun)
        dmyr = dl.derivative(dmr, p_fun, p_test)
        out = self.generate_state()
        dl.assemble(dmyr, tensor=out)
        for bc in self.bc0:
            bc.apply(out)
        return out

    def forSolveAdjIncrementalFwd(self, x, mhat, uhatstar, qoi):
        u_fun = hp.vector2Function(x[STATE], self.Vh[STATE])
        m_fun = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p_fun = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        mhat_fun = hp.vector2Function(mhat, self.Vh[PARAMETER])
        uhatstar_fun = hp.vector2Function(uhatstar, self.Vh[STATE])

        form = self.varf_handler(u_fun, m_fun, p_fun, z_fun)
        u_test = dl.TestFunction(self.Vh[STATE])

        dmr = dl.derivative(form, m_fun, mhat_fun)
        dmxr = dl.derivative(dmr, u_fun, u_test)
        vec_dmxr = self.generate_state()
        dl.assemble(dmxr, tensor=vec_dmxr)
        for bc in self.bc0:
            bc.apply(vec_dmxr)

        dxr = dl.derivative(form, u_fun, uhatstar_fun)
        dxxr = dl.derivative(dxr, u_fun, u_test)
        vec_dxxr = self.generate_state()
        dl.assemble(dxxr, tensor=vec_dxxr)
        for bc in self.bc0:
            bc.apply(vec_dxxr)

        vec_dxxq = self.generate_state()
        qoi.apply_ij(STATE, STATE, uhatstar, vec_dxxq)
        for bc in self.bc0:
            bc.apply(vec_dxxq)

        return vec_dmxr, vec_dxxr, vec_dxxq

    def forSolveAdjAdj(self, x, uhat, uhatstar, mhat, mhatstar, qoi):
        u_fun = hp.vector2Function(x[STATE], self.Vh[STATE])
        m_fun = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p_fun = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        mhat_fun = hp.vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = hp.vector2Function(mhatstar, self.Vh[PARAMETER])
        uhat_fun = hp.vector2Function(uhat, self.Vh[STATE])
        uhatstar_fun = hp.vector2Function(uhatstar, self.Vh[STATE])

        form = self.varf_handler(u_fun, m_fun, p_fun, z_fun)
        p_test = dl.TestFunction(self.Vh[ADJOINT])
        u_test = dl.TestFunction(self.Vh[STATE])

        dxr = dl.derivative(form, u_fun, uhatstar_fun)

        dxxr = dl.derivative(dxr, u_fun, uhat_fun)
        dxxyr = dl.derivative(dxxr, p_fun, p_test)
        vec_dxxyr = self.generate_state()
        dl.assemble(dxxyr, tensor=vec_dxxyr)
        for bc in self.bc0:
            bc.apply(vec_dxxyr)

        dxmr = dl.derivative(dxr, m_fun, mhat_fun)
        dxmyr = dl.derivative(dxmr, p_fun, p_test)
        vec_dxmyr = self.generate_state()
        dl.assemble(dxmyr, tensor=vec_dxmyr)
        for bc in self.bc0:
            bc.apply(vec_dxmyr)

        dyxxr = self.generate_state()
        self.apply_ij(PARAMETER, PARAMETER, mhatstar, dyxxr)

        dmxr = dl.derivative(dxr, u_fun, uhat_fun)
        dxxxr = dl.derivative(dmxr, u_fun, u_test)
        vec_dxxxr = self.generate_state()
        dl.assemble(dxxxr, tensor=vec_dxxxr)
        for bc in self.bc0:
            bc.apply(vec_dxxxr)

        dxmxr = dl.derivative(dxmr, u_fun, u_test)
        vec_dxmxr = self.generate_state()
        dl.assemble(dxmxr, tensor=vec_dxmxr)
        for bc in self.bc0:
            bc.apply(vec_dxmxr)

        dxxxq = self.generate_state()
        qoi.apply_ijk(STATE, STATE, STATE, uhatstar, uhat, dxxxq)
        for bc in self.bc0:
            bc.apply(dxxxq)

        dmr = dl.derivative(form, m_fun, mhat_fun)
        dmxr_ass = dl.derivative(dmr, u_fun, u_test)
        vec_dmxr_ass = self.generate_state()
        dl.assemble(dmxr_ass, tensor=vec_dmxr_ass)
        for bc in self.bc0:
            bc.apply(vec_dmxr_ass)

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmmr = dl.derivative(dmr, m_fun, mhat_fun)
        dmmxr = dl.derivative(dmmr, u_fun, u_test)
        vec_dmmxr = self.generate_state()
        dl.assemble(dmmxr, tensor=vec_dmmxr)
        for bc in self.bc0:
            bc.apply(vec_dmmxr)

        dmyr = dl.derivative(dmr, p_fun, p_test)
        dmyxr = dl.derivative(dmyr, u_fun, u_test)
        vec_dmyxr = self.generate_state()
        dl.assemble(dmyxr, tensor=vec_dmyxr)
        for bc in self.bc0:
            bc.apply(vec_dmyxr)

        dmxr = dl.derivative(dmr, u_fun, uhat_fun)
        dmxxr = dl.derivative(dmxr, u_fun, u_test)
        vec_dmxxr = self.generate_state()
        dl.assemble(dmxxr, tensor=vec_dmxxr)
        for bc in self.bc0:
            bc.apply(vec_dmxxr)

        dxmxq = self.generate_state()
        qoi.apply_ijk(STATE, PARAMETER, STATE, uhatstar, mhat, dxmxq)
        for bc in self.bc0:
            bc.apply(dxmxq)

        dmmxq = self.generate_state()
        qoi.apply_ijk(PARAMETER, PARAMETER, STATE, mhatstar, mhat, dmmxq)
        for bc in self.bc0:
            bc.apply(dmmxq)

        dmxxq = self.generate_state()
        qoi.apply_ijk(PARAMETER, STATE, STATE, mhatstar, uhat, dmxxq)
        for bc in self.bc0:
            bc.apply(dmxxq)

        return (vec_dmxr_ass, vec_dxxyr, vec_dxmyr, vec_dmmxr,
                vec_dmyxr, vec_dmxxr, dxxxq, dxmxq, dmmxq, dmxxq)

    def forSolveAdjFwd(self, x, uhat, uhatstar, mhat, mhatstar, yhat, yhatstar, qoi):
        u_fun = hp.vector2Function(x[STATE], self.Vh[STATE])
        m_fun = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p_fun = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        uhat_fun = hp.vector2Function(uhat, self.Vh[STATE])
        uhatstar_fun = hp.vector2Function(uhatstar, self.Vh[STATE])
        mhat_fun = hp.vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = hp.vector2Function(mhatstar, self.Vh[PARAMETER])
        yhat_fun = hp.vector2Function(yhat, self.Vh[ADJOINT])
        yhatstar_fun = hp.vector2Function(yhatstar, self.Vh[ADJOINT])

        form = self.varf_handler(u_fun, m_fun, p_fun, z_fun)
        z_test = dl.TestFunction(self.Vh[CONTROL])

        dyr = dl.derivative(form, p_fun, yhatstar_fun)
        dyxzr = dl.derivative(dl.derivative(dyr, u_fun, uhat_fun), z_fun, z_test)
        vec_dyxzr = self.generate_control()
        dl.assemble(dyxzr, tensor=vec_dyxzr)

        dymzr = dl.derivative(dl.derivative(dyr, m_fun, mhat_fun), z_fun, z_test)
        vec_dymzr = self.generate_control()
        dl.assemble(dymzr, tensor=vec_dymzr)

        dxr = dl.derivative(form, u_fun, uhatstar_fun)
        dxyzr = dl.derivative(dl.derivative(dxr, p_fun, yhat_fun), z_fun, z_test)
        vec_dxyzr = self.generate_control()
        dl.assemble(dxyzr, tensor=vec_dxyzr)

        dxxzr = dl.derivative(dl.derivative(dxr, u_fun, uhat_fun), z_fun, z_test)
        vec_dxxzr = self.generate_control()
        dl.assemble(dxxzr, tensor=vec_dxxzr)

        dxmzr = dl.derivative(dl.derivative(dxr, m_fun, mhat_fun), z_fun, z_test)
        vec_dxmzr = self.generate_control()
        dl.assemble(dxmzr, tensor=vec_dxmzr)

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmmzr = dl.derivative(dl.derivative(dmr, m_fun, mhat_fun), z_fun, z_test)
        vec_dmmzr = self.generate_control()
        dl.assemble(dmmzr, tensor=vec_dmmzr)

        dmyzr = dl.derivative(dl.derivative(dmr, p_fun, yhat_fun), z_fun, z_test)
        vec_dmyzr = self.generate_control()
        dl.assemble(dmyzr, tensor=vec_dmyzr)

        dmxzr = dl.derivative(dl.derivative(dmr, u_fun, uhat_fun), z_fun, z_test)
        vec_dmxzr = self.generate_control()
        dl.assemble(dmxzr, tensor=vec_dmxzr)

        dmxzq = self.generate_control()
        qoi.apply_ijk(PARAMETER, STATE, CONTROL, mhatstar, uhat, dmxzq)

        dmmzq = self.generate_control()
        qoi.apply_ijk(PARAMETER, PARAMETER, CONTROL, mhatstar, mhat, dmmzq)

        dxxzq = self.generate_control()
        qoi.apply_ijk(STATE, STATE, CONTROL, uhatstar, uhat, dxxzq)

        dxmzq = self.generate_control()
        qoi.apply_ijk(STATE, PARAMETER, CONTROL, uhatstar, mhat, dxmzq)

        dyzr = dl.derivative(form, p_fun, yhatstar_fun)
        dyzr = dl.derivative(dyzr, z_fun, z_test)
        vec_dyzr = self.generate_control()
        dl.assemble(dyzr, tensor=vec_dyzr)

        dxzr = dl.derivative(form, u_fun, uhatstar_fun)
        dxzr = dl.derivative(dxzr, z_fun, z_test)
        vec_dxzr = self.generate_control()
        dl.assemble(dxzr, tensor=vec_dxzr)

        return (
            vec_dyzr,
            vec_dxzr,
            vec_dyxzr,
            vec_dymzr,
            vec_dxyzr,
            vec_dxxzr,
            vec_dxmzr,
            vec_dmmzr,
            vec_dmyzr,
            vec_dmxzr,
            dmxzq,
            dmmzq,
            dxxzq,
            dxmzq,
        )

    def gradientControl(self, x, ustar, pstar, uhat, uhatstar, mhat, mhatstar, phat, phatstar, qoi):
        u_fun = hp.vector2Function(x[STATE], self.Vh[STATE])
        m_fun = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p_fun = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z_fun = hp.vector2Function(x[CONTROL], self.Vh[CONTROL])
        ustar_fun = hp.vector2Function(ustar, self.Vh[STATE])
        pstar_fun = hp.vector2Function(pstar, self.Vh[ADJOINT])
        uhat_fun = hp.vector2Function(uhat, self.Vh[STATE])
        uhatstar_fun = hp.vector2Function(uhatstar, self.Vh[STATE])
        mhat_fun = hp.vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = hp.vector2Function(mhatstar, self.Vh[PARAMETER])
        phat_fun = hp.vector2Function(phat, self.Vh[ADJOINT])
        phatstar_fun = hp.vector2Function(phatstar, self.Vh[ADJOINT])

        form = self.varf_handler(u_fun, m_fun, p_fun, z_fun)
        z_test = dl.TestFunction(self.Vh[CONTROL])

        dyr = dl.derivative(form, p_fun, pstar_fun)
        dyzr = dl.derivative(dyr, z_fun, z_test)
        vec_dyzr = self.generate_control()
        dl.assemble(dyzr, tensor=vec_dyzr)

        dxr = dl.derivative(form, u_fun, ustar_fun)
        dxzr = dl.derivative(dxr, z_fun, z_test)
        vec_dxzr = self.generate_control()
        dl.assemble(dxzr, tensor=vec_dxzr)

        dyr = dl.derivative(form, p_fun, phatstar_fun)
        dyxr = dl.derivative(dyr, u_fun, uhat_fun)
        dyxzr = dl.derivative(dyxr, z_fun, z_test)
        vec_dyxzr = self.generate_control()
        dl.assemble(dyxzr, tensor=vec_dyxzr)

        dymr = dl.derivative(dyr, m_fun, mhat_fun)
        dymzr = dl.derivative(dymr, z_fun, z_test)
        vec_dymzr = self.generate_control()
        dl.assemble(dymzr, tensor=vec_dymzr)

        dxr = dl.derivative(form, u_fun, uhatstar_fun)
        dxyr = dl.derivative(dxr, p_fun, phat_fun)
        dxyzr = dl.derivative(dxyr, z_fun, z_test)
        vec_dxyzr = self.generate_control()
        dl.assemble(dxyzr, tensor=vec_dxyzr)

        dxxr = dl.derivative(dxr, u_fun, uhat_fun)
        dxxzr = dl.derivative(dxxr, z_fun, z_test)
        vec_dxxzr = self.generate_control()
        dl.assemble(dxxzr, tensor=vec_dxxzr)

        dxmr = dl.derivative(dxr, m_fun, mhat_fun)
        dxmzr = dl.derivative(dxmr, z_fun, z_test)
        vec_dxmzr = self.generate_control()
        dl.assemble(dxmzr, tensor=vec_dxmzr)

        dmr = dl.derivative(form, m_fun, mhatstar_fun)
        dmmr = dl.derivative(dmr, m_fun, mhat_fun)
        dmmzr = dl.derivative(dmmr, z_fun, z_test)
        vec_dmmzr = self.generate_control()
        dl.assemble(dmmzr, tensor=vec_dmmzr)

        dmyr = dl.derivative(dmr, p_fun, phat_fun)
        dmyzr = dl.derivative(dmyr, z_fun, z_test)
        vec_dmyzr = self.generate_control()
        dl.assemble(dmyzr, tensor=vec_dmyzr)

        dmxr = dl.derivative(dmr, u_fun, uhat_fun)
        dmxzr = dl.derivative(dmxr, z_fun, z_test)
        vec_dmxzr = self.generate_control()
        dl.assemble(dmxzr, tensor=vec_dmxzr)

        dmxzq = self.generate_control()
        qoi.apply_ijk(PARAMETER, STATE, CONTROL, mhatstar, uhat, dmxzq)

        dmmzq = self.generate_control()
        qoi.apply_ijk(PARAMETER, PARAMETER, CONTROL, mhatstar, mhat, dmmzq)

        dxxzq = self.generate_control()
        qoi.apply_ijk(STATE, STATE, CONTROL, uhatstar, uhat, dxxzq)

        dxmzq = self.generate_control()
        qoi.apply_ijk(STATE, PARAMETER, CONTROL, uhatstar, mhat, dxmzq)

        return (
            vec_dyzr,
            vec_dxzr,
            vec_dyxzr,
            vec_dymzr,
            vec_dxyzr,
            vec_dxxzr,
            vec_dxmzr,
            vec_dmmzr,
            vec_dmyzr,
            vec_dmxzr,
            dmxzq,
            dmmzq,
            dxxzq,
            dxmzq,
        )
