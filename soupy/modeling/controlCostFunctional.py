import logging 
import numpy as np
import dolfin as dl

from .variables import STATE, PARAMETER, ADJOINT, CONTROL

class ControlCostFunctional:
    """
    This class defines the cost function for solving an optimal control problem
    under uncertainty.
    """

    def objective(self, z):
        raise NotImplementedError("Child class should implement method objective")

    def cost(self, z):
        """
        Given the control variable z evaluate the cost functional
        """
        raise NotImplementedError("Child class should implement method costValue")

    def costGrad(self, z):
        """
        Given the control variable z evaluate the gradient of the cost functional
        """
        raise NotImplementedError("Child class should implement method costGrad")

    def costHessian(self, z, dz):
        """
        Given the control variable z evaluate the Hessian of the cost functional
        acting in direction dz
        """
        raise NotImplementedError("Child class should implement method costHess")


class DeterministicControlCostFunctional(ControlCostFunctional):
    """
    This class implements a deterministic approximation for the optimal control problem
    under uncertainty by considering a fixed parameter at the mean of the prior
    """

    def __init__(self, model, prior, penalization=None):
        self.model = model
        self.prior = prior
        self.penalization = penalization

        self.u, self.m, self.p, self.z = self.model.generate_vector(component="ALL")
        self.m.zero()
        self.m.axpy(1.0, self.prior.mean)
        self.x = [self.u, self.m, self.p, self.z]
        self.diff_helper = self.model.generate_vector(CONTROL)
        self.grad_penalization = self.model.generate_vector(CONTROL)
        self.grad_objective = self.model.generate_vector(CONTROL)
        self.q_bar = 0 

        self.comm = self.u.mpi_comm()
        self.mpi_size = self.comm.Get_size()

        self.rhs_fwd = self.model.generate_vector(STATE)
        self.rhs_adj = self.model.generate_vector(ADJOINT)
        self.rhs_adj2 = self.model.generate_vector(ADJOINT)
        self.uhat = self.model.generate_vector(STATE)
        self.phat = self.model.generate_vector(ADJOINT)
        self.zhelp = self.model.generate_vector(CONTROL)

        self.has_forward_solve = False 
        self.has_adjoint_solve = False 

    def generate_vector(self, component="ALL"):
        return self.model.generate_vector(component)

    def objective(self, z):
        self.z.zero()
        self.z.axpy(1.0, z)
        self.model.solveFwd(self.x[STATE], self.x)
        return self.model.cost(self.x)

    def computeComponents(self, z, order=0):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `z` the control variable 
            - :code: `order` the order of derivatives needed. 
                    0 for cost. 1 for grad. 2 for Hessian
        """

        # Check if a new control variable is used 
        new_forward_solve = False 
        self.diff_helper.zero()
        self.diff_helper.axpy(1.0, self.z)
        self.diff_helper.axpy(-1.0, z)
        diff_norm = np.sqrt(self.diff_helper.inner(self.diff_helper))
        
        # Check if new forward solve is needed 
        if diff_norm > dl.DOLFIN_EPS or not self.has_forward_solve:
            # Update control variable (changes all samples)
            # Ask that new forward and adjoint solves are computed 
            self.z.zero()
            self.z.axpy(1.0, z)
            # new_forward_solve = True
            logging.info("Using new forward solve")
            self.model.solveFwd(self.u, self.x) 
            
            if order >= 1: 
                logging.info("Using new adjoint solve")
                self.model.solveAdj(self.p, self.x)
                self.model.evalGradientControl(self.x, self.grad_objective)
                self.has_adjoint_solve = True
            else:
                self.has_adjoint_solve = False

        elif order >= 1 and not self.has_adjoint_solve:
            logging.info("Using new adjoint solve")
            self.model.solveAdj(self.p, self.x)
            self.model.evalGradientControl(self.x, self.grad_objective)
            self.has_adjoint_solve = True


    def cost(self, z, order=0, **kwargs):
        self.computeComponents(z, order=order)
        objective = self.model.cost(self.x)
        if self.penalization is None:
            penalization = 0.0
        else:
            penalization = self.penalization.cost(z)
        return objective + penalization 

    def costGrad(self, z, out):
        """
        Compute the gradient
        Assumes self.costValue(z, order=1) has been called 
        """
        self.model.evalGradientControl(self.x, out)
        if self.penalization is not None:
            self.penalization.grad(z, self.grad_penalization)
            out.axpy(1.0, self.grad_penalization)

        gradnorm = np.sqrt(out.inner(out))
        return gradnorm

    def costHessian(self, z, zhat, out):
        """
        Apply the the reduced Hessian to the vector :code:`zhat`
        evaluated at control variable :code:`z`
        Return the result in :code:`out`.
        Need to call self.costValue(self, order) with order>=1 before self.costHessian
        """
        self.model.setPointForHessianEvaluations(self.x)
        self.model.applyCz(zhat, self.rhs_fwd)
        self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWuz(zhat, self.rhs_adj2)
        self.rhs_adj.axpy(-1., self.rhs_adj2)
        self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        self.model.applyWzz(zhat, out)

        self.model.applyCzt(self.phat, self.zhelp)
        out.axpy(1., self.zhelp)
        self.model.applyWzu(self.uhat, self.zhelp)
        out.axpy(-1., self.zhelp)

        if self.penalization is not None:
            Hzhat_penalization = self.model.generate_vector(CONTROL)
            self.penalization.hessian(z, zhat, Hzhat_penalization)
            out.axpy(1.0, Hzhat_penalization)




class RiskMeasureControlCostFunctional:
    def __init__(self, risk_measure, penalization=None):
        self.risk_measure = risk_measure
        self.penalization = penalization
        self.grad_risk = self.risk_measure.generate_vector(CONTROL)
        self.grad_penalization = self.risk_measure.generate_vector(CONTROL)

    def generate_vector(self, component="ALL"):
        return self.risk_measure.generate_vector(component)

    def cost(self, z, order=0, **kwargs):
        self.risk_measure.computeComponents(z, order=order, **kwargs)
        cost_risk = self.risk_measure.cost()
        if self.penalization is not None:
            cost_penalization = self.penalization.cost(z)
        else:
            cost_penalization = 0.0

        return cost_risk+cost_penalization

    def costGrad(self, z, out):
        """
        First calls cost with order = 1
        """
        out.zero()
        self.risk_measure.computeComponents(z, order=1)

        # Risk measure gradient
        self.risk_measure.costGrad(self.grad_risk)
        out.axpy(1.0, self.grad_risk)

        if self.penalization is not None:
            self.penalization.grad(z, self.grad_penalization)
            out.axpy(1.0, self.grad_penalization)

        gradnorm = np.sqrt(out.inner(out))
        return gradnorm

    def costHessian(self, z, zhat, out):
        pass




