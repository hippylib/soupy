import logging 
import numpy as np 
import dolfin as dl 
import mpi4py

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
from hippylib import ParameterList, Random

from .riskMeasure import RiskMeasure
from .variables import STATE, PARAMETER, ADJOINT, CONTROL

from ..collectives import NullCollective, MultipleSamePartitioningPDEsCollective, MultipleSerialPDEsCollective


def meanVarRiskMeasureSAASettings(data = {}):
    # This should be a Parameter
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0,'Weighting factor for variance']
    data['seed'] = [1, 'rng seed for sampling']

    return ParameterList(data)


def _allocate_sample_sizes(sample_size, comm_sampler):
    """
    Compute the number of samples needed in each process 
    Return result as a list 
    """ 
    n, r = divmod(sample_size, comm_sampler.size)
    sample_size_allprocs = []
    for i_rank in range(comm_sampler.size):
        if i_rank < r: 
            sample_size_allprocs.append(n+1)
        else:
            sample_size_allprocs.append(n)
    return sample_size_allprocs


class MeanVarRiskMeasureSAA_MPI(RiskMeasure):
    """
    Class for memory efficient evaluation of the Mean + Variance risk measure 
    E[X] + beta Var[X]. 
    """

    def __init__(self, control_model, prior, settings = meanVarRiskMeasureSAASettings(), comm_sampler=mpi4py.MPI.COMM_WORLD):
        """
        Parameters
            - :code: `control_model` control model of problem 
            - :code: `prior` prior for uncertain parameter
            - :code: `settings` additional settings
            - :code: `comm_sampler` MPI communicator for the sampling parallelism 
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.sample_size = self.settings['sample_size']
        self.beta = settings['beta']

        assert self.sample_size >= comm_sampler.Get_size()
        self.comm_sampler = comm_sampler 
        self.sample_size_allprocs = _allocate_sample_sizes(self.sample_size, self.comm_sampler)
        self.sample_size_proc = self.sample_size_allprocs[self.comm_sampler.rank]

        if comm_sampler.Get_size() == 1:
            self.collective = NullCollective()
        else:
            self.collective = MultipleSerialPDEsCollective(self.comm_sampler)

        # Within process 
        self.g = self.model.generate_vector(CONTROL)
        self.q_samples = np.zeros(self.sample_size_proc)

        # self.q_bar_proc = np.array([0.])
        # self.g_bar_proc = self.model.generate_vector(CONTROL)
        # self.qg_bar_proc = self.model.generate_vector(CONTROL)

        # Aggregate components for computing cost, grad, hess
        self.q_bar = 0
        self.q2_bar = 0 
        self.g_bar = self.model.generate_vector(CONTROL)
        self.qg_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector(self.model.problem.Vh[STATE].mesh().mpi_comm()) # use the mesh mpi comm 
        self.prior.init_vector(self.noise, "noise")
        rng = Random(seed=self.settings['seed'])

        # Generate samples for m 
        self.x_mc = [] 
        self.z = self.model.generate_vector(CONTROL)

        if mpi4py.MPI.COMM_WORLD.rank == 0:
            logging.info("Initial sampling of stochastic parameter")

        # Burn in for parallel sampling  
        n_burn = int(np.sum(self.sample_size_allprocs[:self.comm_sampler.rank]))
        for i in range(n_burn):
            rng.normal(1.0, self.noise)
        
        # Actual sampling 
        for i in range(self.sample_size_proc):
            ui = self.model.generate_vector(STATE)
            mi = self.model.generate_vector(PARAMETER)
            pi = self.model.generate_vector(ADJOINT) 
            x = [ui, mi, pi, self.z]
            rng.normal(1.0, self.noise)
            self.prior.sample(self.noise, mi)
            self.x_mc.append(x)

        if mpi4py.MPI.COMM_WORLD.rank == 0:
            logging.info("Done sampling")

        # Helper variable 
        self.diff_helper= self.model.generate_vector(CONTROL)
        self.has_adjoint_solve = False
        self.has_forward_solve = False

    def generate_vector(self, component = "ALL"):
        return self.model.generate_vector(component)

    def computeComponents(self, z, order=0, **kwargs):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `z` the control variable 
            - :code: `order` the order of derivatives needed. 
                    0 for cost. 1 for grad. 2 for Hessian
            - :code: `**kwargs` dummy keyword arguments for compatibility 
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
            new_forward_solve = True
            logging.info("Using new forward solve")

        # Check if a new adjoint solve is needed 
        if new_forward_solve:
            # Reset the averages for new solve 
            self.q_bar = 0 
            self.q2_bar = 0 

            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 
            # print("Solving new adjoint")
            self.g_bar.zero()
            self.qg_bar.zero()

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
        else:
            # If we don't already have an adjoint, compute a new one 
            # print("Solving new adjoint")
            new_adjoint_solve = True 
            self.g_bar.zero()
            self.qg_bar.zero()
            # In this case, self.has_adjoint_solve is already False

        # Now actually compute the solves
        if new_forward_solve or new_adjoint_solve:
            for i in range(self.sample_size_proc):
                x = self.x_mc[i]

                if new_forward_solve:
                    # Solve state           
                    self.model.solveFwd(x[STATE], x)
                    
                    # Compute the averages
                    qi = self.model.cost(x) 
                    self.q_samples[i] = qi 
                    self.q_bar += qi/self.sample_size
                    self.q2_bar += qi**2/self.sample_size

                if order >= 1 and new_adjoint_solve:
                    self.model.solveAdj(x[ADJOINT], x)
                    self.model.evalGradientControl(x, self.g)
                    self.g_bar.axpy(1/self.sample_size, self.g)
                    self.qg_bar.axpy(self.q_samples[i]/self.sample_size, self.g)
            
            if new_forward_solve:
                self.q_bar = self.collective.allReduce(self.q_bar, "SUM")
                self.q2_bar = self.collective.allReduce(self.q2_bar, "SUM")
            
            if order >= 1 and new_adjoint_solve:
                self.collective.allReduce(self.g_bar, "SUM")
                self.collective.allReduce(self.qg_bar, "SUM")

                # We have computed a new adjoint solve 
                # Don't need to alter if it's any other case 
                self.has_adjoint_solve = True 

            # Will always be true after solving 
            self.has_forward_solve = True 
    
    def cost(self):
        """
        Evaluates the cost given by the risk measure
        Assumes :code: `computeComponents` has been called
        """
        return self.q_bar + self.beta * (self.q2_bar - self.q_bar**2)

    def costGrad(self, g):
        """
        Evaluates the gradient by the risk measure
        Assumes :code: `computeComponents` has been called with :code: `order>=1`
        Parameters
            - :code: `g` output vector for the gradient
        """
        # print("(proc %d) q_bar = %g" %(self.comm_sampler.Get_rank(), self.q_bar))
        g.zero()
        g.axpy(1.0, self.g_bar)
        g.axpy(2*self.beta, self.qg_bar)
        g.axpy(-2*self.beta*self.q_bar, self.g_bar)

    def costHessian(self, zhat, Hzhat):
        logging.warning("No hessian implemented")
        return 

        # self.model.setPointForHessianEvaluations(self.x)
        # self.model.applyCz(zhat, self.rhs_fwd)
        # self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        # self.model.applyWuu(self.uhat, self.rhs_adj)
        # self.model.applyWuz(zhat, self.rhs_adj2)
        # self.rhs_adj.axpy(-1., self.rhs_adj2)
        # self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        # self.model.applyWzz(zhat, Hzhat)

        # self.model.applyCzt(self.phat, self.zhelp)
        # Hzhat.axpy(1., self.zhelp)
        # self.model.applyWzu(self.uhat, self.zhelp)
        # Hzhat.axpy(-1., self.zhelp)

    def gatherSamples(self, root=0):
        q_all = None 
        if self.comm_sampler.Get_rank() == 0:
            q_all = np.zeros(self.sample_size)
        self.comm_sampler.Gatherv(self.q_samples, q_all, root=root)
        return q_all 


class MeanVarRiskMeasureSAA(RiskMeasure):
    """
    Class for memory efficient evaluation of the Mean + Variance risk measure 
    E[X] + beta Var[X]. 
    """

    def __init__(self, control_model, prior, settings = meanVarRiskMeasureSAASettings()):
        """
        Parameters
            - :code: `control_model` control model of problem 
            - :code: `prior` prior for uncertain parameter
            - :code: `settings` additional settings
        """
        self.model = control_model
        self.prior = prior
        self.settings = settings
        self.sample_size = self.settings['sample_size']
        self.beta = settings['beta']

        # Aggregate components for computing cost, grad, hess
        self.g = self.model.generate_vector(CONTROL)
        self.q_samples = np.zeros(self.sample_size)

        self.q_bar = 0
        self.g_bar = self.model.generate_vector(CONTROL)
        self.qg_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector()
        self.prior.init_vector(self.noise, "noise")
        rng = Random(seed=self.settings['seed'])

        # Generate samples for m 
        self.x_mc = [] 
        self.z = self.model.generate_vector(CONTROL)
        logging.info("Initial sampling of stochastic parameter")
        for i in range(self.sample_size):
            ui = self.model.generate_vector(STATE)
            mi = self.model.generate_vector(PARAMETER)
            pi = self.model.generate_vector(ADJOINT) 
            x = [ui, mi, pi, self.z]
            rng.normal(1.0, self.noise)
            self.prior.sample(self.noise, mi)
            self.x_mc.append(x)

        # Helper variable 
        self.diff_helper= self.model.generate_vector(CONTROL)
        self.has_adjoint_solve = False
        self.has_forward_solve = False

    def generate_vector(self, component = "ALL"):
        return self.model.generate_vector(component)

    def computeComponents(self, z, order=0, **kwargs):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `z` the control variable 
            - :code: `order` the order of derivatives needed. 
                    0 for cost. 1 for grad. 2 for Hessian
            - :code: `**kwargs` dummy keyword arguments for compatibility 
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
            new_forward_solve = True
            logging.info("Using new forward solve")

        # Check if a new adjoint solve is needed 
        if new_forward_solve:
            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 
            self.g_bar.zero()
            self.qg_bar.zero()

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
        else:
            # If we don't already have an adjoint, compute a new one 
            new_adjoint_solve = True 
            # In this case, self.has_adjoint_solve is already False
            self.g_bar.zero()
            self.qg_bar.zero()


        if new_forward_solve or new_adjoint_solve:
        # Now actually compute the solves
            for i in range(self.sample_size):
                x = self.x_mc[i]
                    # Solve state           
                if new_forward_solve:
                    self.model.solveFwd(x[STATE], x)
                    self.q_samples[i] = self.model.cost(x)

                if order >= 1 and new_adjoint_solve:
                    self.model.solveAdj(x[ADJOINT], x)
                    self.model.evalGradientControl(x, self.g)
                    self.g_bar.axpy(1/self.sample_size, self.g)
                    self.qg_bar.axpy(self.q_samples[i]/self.sample_size, self.g)

            self.q_bar = np.mean(self.q_samples)

            if order >= 1: 
                # We have computed a new adjoint solve 
                # Don't need to alter if it's any other case 
                self.has_adjoint_solve = True 

            # Will always be true after solving 
            self.has_forward_solve = True 
    
    def cost(self):
        """
        Evaluates the cost given by the risk measure
        Assumes :code: `computeComponents` has been called
        """
        return self.q_bar + self.beta * np.std(self.q_samples)**2

    def costGrad(self, g):
        """
        Evaluates the gradient by the risk measure
        Assumes :code: `computeComponents` has been called with :code: `order>=1`
        Parameters
            - :code: `g` output vector for the gradient
        """
        g.zero()
        g.axpy(1.0, self.g_bar)
        g.axpy(2*self.beta, self.qg_bar)
        g.axpy(-2*self.beta*self.q_bar, self.g_bar)

    def costHessian(self, zhat, Hzhat):
        logging.warning("No hessian implemented")
        return 

        # self.model.setPointForHessianEvaluations(self.x)
        # self.model.applyCz(zhat, self.rhs_fwd)
        # self.model.solveFwdIncremental(self.uhat, self.rhs_fwd)
        # self.model.applyWuu(self.uhat, self.rhs_adj)
        # self.model.applyWuz(zhat, self.rhs_adj2)
        # self.rhs_adj.axpy(-1., self.rhs_adj2)
        # self.model.solveAdjIncremental(self.phat, self.rhs_adj)
        # self.model.applyWzz(zhat, Hzhat)

        # self.model.applyCzt(self.phat, self.zhelp)
        # Hzhat.axpy(1., self.zhelp)
        # self.model.applyWzu(self.uhat, self.zhelp)
        # Hzhat.axpy(-1., self.zhelp)