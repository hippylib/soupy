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
from .smoothPlusApproximation import SmoothPlusApproximationQuartic, SmoothPlusApproximationSoftplus
from .augmentedVector import AugmentedVector



def sampleSuperquantile(samples, beta):
    """
    Evaluate superquantile from samples 
    """ 
    quantile = np.quantile(samples, beta) 
    return quantile + np.mean(np.maximum(samples - quantile, 0))/(1-beta)
    


def superquantileRiskMeasureSAASettings(data = {}):
    # This should be a Parameter
    data['sample_size'] = [100,'Number of Monte Carlo samples']
    data['beta'] = [0.95, 'Quantile value for superquantile']
    data['epsilon'] = [0.01, 'Sharpness of smooth plus approximation']
    data['seed'] = [1, 'rng seed for sampling']
    data['smoothplus_type'] = ['quartic', 'approximation type for smooth plus function']
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


class SuperquantileRiskMeasureSAA_MPI(RiskMeasure):
    """
    Class for memory efficient evaluation of the Mean + Variance risk measure 
    E[X] + beta Var[X]. 
    """

    def __init__(self, control_model, prior, settings = superquantileRiskMeasureSAASettings(), comm_sampler=mpi4py.MPI.COMM_WORLD):
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
        if self.settings["smoothplus_type"] == "softplus":
            self.smoothplus = SmoothPlusApproximationSoftplus(self.settings["epsilon"])
        elif self.settings["smoothplus_type"] == "quartic":
            self.smoothplus = SmoothPlusApproximationQuartic(self.settings["epsilon"])
        else:
            # Default case 
            self.smoothplus = SmoothPlusApproximationQuartic(self.settings["epsilon"])

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

        # Aggregate components for computing cost, grad, hess
        self.s_bar = 0 
        self.sprime_bar = 0 
        self.sprime_g_bar = self.model.generate_vector(CONTROL)

        # For sampling
        self.noise = dl.Vector(self.model.problem.Vh[STATE].mesh().mpi_comm()) # use the mesh mpi comm 
        self.prior.init_vector(self.noise, "noise")
        rng = Random(seed=self.settings['seed'])

        # Generate samples for m 
        self.x_mc = [] 
        self.z = self.model.generate_vector(CONTROL)
        self.zt = AugmentedVector(self.z, copy_vector=False)

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
        if component == CONTROL:
            return AugmentedVector(self.model.generate_vector(CONTROL), copy_vector=False)
        else:
            return self.model.generate_vector(component)

    def computeComponents(self, zt, order=0, **kwargs):
        """
        Computes the components for the stochastic approximation of the cost
        Parameters:
            - :code: `z` the control variable 
            - :code: `order` the order of derivatives needed. 
                    0 for cost. 1 for grad. 2 for Hessian
            - :code: `**kwargs` dummy keyword arguments for compatibility 
        """
        z = zt.get_vector()
        t = zt.get_scalar()
        new_forward_solve = False 

        self.s_bar = 0 

        # Check if a new control variable is used 
        self.diff_helper.zero()
        self.diff_helper.axpy(1.0, self.z)
        self.diff_helper.axpy(-1.0, z)
        diff_norm = np.sqrt(self.diff_helper.inner(self.diff_helper))
        
        # Store zt in class 
        self.zt.zero()
        self.zt.axpy(1.0, zt)

        # Check if new forward solve is needed 
        if diff_norm > dl.DOLFIN_EPS or not self.has_forward_solve:
            # Update control variable (changes all samples)
            # Ask that new forward and adjoint solves are computed 
            new_forward_solve = True
            logging.info("Using new forward solve")

        # Check if a new adjoint solve is needed 
        if new_forward_solve:
            # Reset the averages for new solve 
            # Make sure previous adjoint solve is no longer valid 
            self.has_adjoint_solve = False 
            new_adjoint_solve = True 
            self.sprime_bar = 0 
            self.sprime_g_bar.zero()

        elif self.has_adjoint_solve:
            # If it's not a new forward solve, check if we already have an adjoint 
            new_adjoint_solve = False 
        else:
            # If we don't already have an adjoint, compute a new one 
            new_adjoint_solve = True 
            self.sprime_bar = 0 
            self.sprime_g_bar.zero()
            # In this case, self.has_adjoint_solve is already False

        # Now actually compute the solves
        for i in range(self.sample_size_proc):
            x = self.x_mc[i]

            if new_forward_solve:
                # Solve state           
                self.model.solveFwd(x[STATE], x)
                
            # Compute the averages
            qi = self.model.cost(x) 
            self.q_samples[i] = qi 
            self.s_bar += self.smoothplus(qi - t)/self.sample_size

            if order >= 1 and new_adjoint_solve:
                self.model.solveAdj(x[ADJOINT], x)
                self.model.evalGradientControl(x, self.g)
                self.sprime_bar += self.smoothplus.grad(qi - t)/self.sample_size
                self.sprime_g_bar.axpy(self.smoothplus.grad(qi - t)/self.sample_size, self.g)
        
        self.s_bar = self.collective.allReduce(self.s_bar, "SUM")
        
        if order >= 1 and new_adjoint_solve:
            self.collective.allReduce(self.sprime_g_bar, "SUM")
            self.sprime_bar = self.collective.allReduce(self.sprime_bar, "SUM")

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
        t = self.zt.get_scalar()
        return  t + 1/(1-self.beta) * self.s_bar

    def costGrad(self, gt):
        """
        Evaluates the gradient by the risk measure
        Assumes :code: `computeComponents` has been called with :code: `order>=1`
        Parameters
            - :code: `g` output vector as an augmented vector 
        """
        # print("(proc %d) q_bar = %g" %(self.comm_sampler.Get_rank(), self.q_bar))
        dzJ_np = self.sprime_g_bar.get_local()/(1-self.beta)
        dtJ_np = 1 - self.sprime_bar/(1-self.beta)
        dz_np = np.append(dzJ_np, dtJ_np)
        gt.set_local(dz_np) 

    def costHessian(self, zhat, Hzhat):
        logging.warning("No hessian implemented")
        return 

    def gatherSamples(self, root=0):
        q_all = None 
        if self.comm_sampler.Get_rank() == 0:
            q_all = np.zeros(self.sample_size)
        self.comm_sampler.Gatherv(self.q_samples, q_all, root=root)
        return q_all 

    def superquantile(self):
        """ 
        Evaluate the superquantile using the computed samples 
        """
        q_all = self.gatherSamples(root=0)
        value = 0.0
        if self.comm_sampler.Get_rank() == 0: 
            value = sampleSuperquantile(q_all, self.beta)
        value = self.comm_sampler.bcast(value, root=0)
        return value 



