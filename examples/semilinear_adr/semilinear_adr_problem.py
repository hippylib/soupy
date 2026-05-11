import pickle

from dataclasses import dataclass

import dolfin as dl
import numpy as np
from mpi4py import MPI

import hippylib as hp

try:
    import taylorGM
except ImportError:
    taylorGM = None


@dataclass
class MeshParameters:
    """
    Parameters for the mesh 
    """
    nx : int = 128
    ny : int = 128


@dataclass
class PDEParameters:
    """
    Parameters of the PDE model 
    """
    reaction: float = 0.01
    velocity: tuple = (0.1, 0.1)

    source_strength: float = 10.0
    source_width: float = 0.2
    source_loc: tuple = (0.25, 0.5)


@dataclass
class PriorParameters:
    """
    Parameters for the prior 
    """
    mean : float = 0.0
    correlation_length : float = 1.0
    pointwise_variance : float = 1.0
    robin_bc : bool=True

    def compute_prior_coefficients(self):
        """
        Computes the coefficient for the BiLaplacian prior 
        """
        gamma, delta = hp.BiLaplacianComputeCoefficients(self.pointwise_variance, self.correlation_length, 2)
        return gamma, delta 


def save_parameters(save_dir, mesh_parameters, pde_parameters, prior_parameters, qoi_type):
    """
    Pickle the parameters for making the problem 
    """
    parameters_all = dict()
    parameters_all['mesh'] = mesh_parameters
    parameters_all['pde'] = pde_parameters
    parameters_all['prior'] = prior_parameters
    parameters_all['qoi'] = qoi_type
    with open("%s/parameters.p" %(save_dir), "wb") as parameter_file:
        pickle.dump(parameters_all, parameter_file)



def load_adr_problem(save_dir, comm_mesh):
    """
    Load parameters from a save file and make the corresponding 
    ADR problem 
    """

    with open("%s/parameters.p" %(save_dir), "rb") as parameter_file:
        parameter_all = pickle.load(parameter_file)
    
    mesh_parameters = parameter_all['mesh']
    pde_parameters = parameter_all['pde']
    prior_parameters = parameter_all['prior']
    qoi_type = parameter_all['qoi']

    Vh, pde, prior, qoi = setup_adr_problem(mesh_parameters, 
            pde_parameters, prior_parameters, qoi_type, comm_mesh)
    return Vh, pde, prior, qoi


def setup_adr_problem(mesh_parameters, 
        pde_parameters, 
        prior_parameters, 
        qoi_type, 
        comm_mesh):
    """
    Setup the ADR problem 
    """
    mesh = setup_mesh(mesh_parameters, comm_mesh)
    Vh = setup_function_spaces(mesh)
    pde = setup_pde(Vh, pde_parameters)
    prior = setup_prior(Vh, prior_parameters)
    qoi = setup_qoi(Vh, qoi_type, mesh)
    return Vh, pde, prior, qoi 



def setup_mesh(mesh_parameters, comm_mesh):
    mesh = dl.UnitSquareMesh(comm_mesh, mesh_parameters.nx, mesh_parameters.ny)
    return mesh 



def gaussian2DExpression(comm_mesh, strength, center, width, const=0, degree=5):
    gauss = dl.Expression("c + a * exp(-(pow(x[0]-x0, 2) + pow(x[1]-x1, 2))/(2*b*b))", 
            c=const,
            a = strength,
            b = width,
            x0 = center[0],
            x1 = center[1],
            degree=degree,
            mpi_comm=comm_mesh)
    return gauss



def setup_function_spaces(mesh):
    """
    Setup the function spaces from the mesh 
    """
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]
    return Vh 



def setup_pde(Vh, pde_parameters):
    """
    Set up the semilinear adr PDE
    """
    # Setup left boundary conditions 
    bc = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), "on_boundary && near(x[0], 0.0)")
    bc0 = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), "on_boundary && near(x[0], 0.0)")
    pde_varf = SemilinearEllipticVarfHandler(Vh, pde_parameters)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)
    return pde 


def setup_prior(Vh, prior_parameters):
    """
    Setup the prior from the function space and prior parameters
    """
    m_mean = dl.interpolate(dl.Constant(prior_parameters.mean), Vh[hp.PARAMETER])
    gamma, delta = prior_parameters.compute_prior_coefficients()
    prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], 
            gamma, 
            delta, 
            mean=m_mean.vector(), 
            robin_bc=prior_parameters.robin_bc
    )
    # if taylorGM is not None and hasattr(taylorGM, "SqrtPrecisionPDEGaussian"):
    #     return taylorGM.SqrtPrecisionPDEGaussian(prior)
    return prior


class SemilinearEllipticVarfHandler:
    """
    Variational form for the semilinear elliptic PDE 
    """
    def __init__(self, Vh, parameters):
        self.Vh = Vh 
        self.mpi_comm = self.Vh[hp.STATE].mesh().mpi_comm()
        self.parameters = parameters
        self.reaction = dl.Constant(self.parameters.reaction)
        self.velocity = dl.Constant(self.parameters.velocity)
        self.source = gaussian2DExpression(self.mpi_comm, self.parameters.source_strength, 
                self.parameters.source_loc, self.parameters.source_width)

    def __call__(self, u, m, p):
        varf = dl.inner(dl.exp(m) * dl.grad(u), dl.grad(p)) * dl.dx \
                + dl.inner(self.velocity, dl.grad(u)) * p * dl.dx \
                + self.reaction*u**3*p * dl.dx \
                - self.source*p * dl.dx
        return varf


def _l2_norm(u, m):
    return u**2 * dl.dx 


def _energy_norm(u, m):
    return dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u))*dl.dx 


def _cubic(u, m):
    return u**3 * dl.dx 


def _left_boundary(x, on_boundary):
    return on_boundary and dl.near(x[1], 0.0)


class BoundaryFluxVarf:
    def __init__(self, mesh):
        self.mesh = mesh

        boundaries = dl.MeshFunction("size_t", self.mesh, self.mesh.geometry().dim() - 1)
        boundaries.set_all(0)
        left = dl.AutoSubDomain(_left_boundary)
        left.mark(boundaries, 1)

        self.ds = dl.Measure("ds", domain=self.mesh, subdomain_data=boundaries)
        self.n = dl.FacetNormal(self.mesh)

    def __call__(self, u, m):
        form = - dl.inner(dl.exp(m) * dl.grad(u), self.n) * self.ds(1)
        return form 


def setup_qoi(Vh, qoi_type, mesh):
    """
    Setup the quantity of interest from the function space and observation parameters
    """
    if qoi_type == "l2":
        qoi_varf = _l2_norm
    elif qoi_type == "energy":
        qoi_varf = _energy_norm
    elif qoi_type == "cubic":
        qoi_varf = _cubic
    elif qoi_type == 'flux':
        qoi_varf = BoundaryFluxVarf(mesh)
    else:
        raise ValueError("Unsupported qoi_type")
    qoi = hp.VariationalQoi(Vh, qoi_varf)
    return qoi 

