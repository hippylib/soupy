import math
import pickle

from dataclasses import dataclass

import dolfin as dl
import numpy as np
from mpi4py import MPI

import hippylib as hp
import soupy

try:
    import taylorGM
except ImportError:
    taylorGM = None


@dataclass
class MeshParameters:
    """Parameters for the mesh."""

    nx: int = 128
    ny: int = 128


@dataclass
class PDEParameters:
    """Parameters of the PDE model."""

    reaction: float = 0.1


@dataclass
class PriorParameters:
    """Parameters for the GRF prior."""

    mean: float = -1.0
    gamma: float = 0.1
    delta: float = 5.0
    robin_bc: bool = True


@dataclass
class ControlParameters:
    """Parameters for the Gaussian well control basis."""

    n_wells_per_side: int = 7
    loc_lower: float = 0.1
    loc_upper: float = 0.9
    well_width: float = 0.08

    @property
    def n_control(self) -> int:
        return self.n_wells_per_side ** 2


def save_parameters(save_dir, mesh_parameters, pde_parameters, prior_parameters, qoi_type):
    """Pickle the parameters for making the problem."""
    parameters_all = dict()
    parameters_all['mesh'] = mesh_parameters
    parameters_all['pde'] = pde_parameters
    parameters_all['prior'] = prior_parameters
    parameters_all['qoi'] = qoi_type
    with open("%s/parameters.p" % (save_dir), "wb") as parameter_file:
        pickle.dump(parameters_all, parameter_file)



def load_adr_problem(save_dir, comm_mesh):
    """Load parameters from a save file and make the corresponding ADR problem."""
    with open("%s/parameters.p" % (save_dir), "rb") as parameter_file:
        parameter_all = pickle.load(parameter_file)

    mesh_parameters = parameter_all['mesh']
    pde_parameters = parameter_all['pde']
    prior_parameters = parameter_all['prior']
    qoi_type = parameter_all['qoi']

    Vh, pde, prior, qoi = setup_adr_problem(
        mesh_parameters, pde_parameters, prior_parameters, qoi_type, comm_mesh
    )
    return Vh, pde, prior, qoi



def setup_adr_problem(mesh_parameters, pde_parameters, prior_parameters, qoi_type, comm_mesh):
    """Setup the ADR problem."""
    mesh = setup_mesh(mesh_parameters, comm_mesh)
    Vh = setup_function_spaces(mesh)
    pde = setup_pde(Vh, pde_parameters)
    prior = setup_prior(Vh, prior_parameters)
    qoi = setup_qoi(Vh, qoi_type, mesh)
    return Vh, pde, prior, qoi



def setup_mesh(mesh_parameters, comm_mesh):
    return dl.UnitSquareMesh(comm_mesh, mesh_parameters.nx, mesh_parameters.ny)



def setup_function_spaces(mesh):
    """Setup the state/parameter/adjoint spaces from the mesh."""
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    return [Vh_STATE, Vh_PARAMETER, Vh_STATE]



def setup_control_function_space(mesh, control_parameters: ControlParameters):
    """Return the finite-dimensional Gaussian-well coefficient space."""
    return dl.VectorFunctionSpace(mesh, "R", degree=0, dim=control_parameters.n_control)



def control_well_centers(control_parameters: ControlParameters) -> np.ndarray:
    """Return the Gaussian well centers in the same ordering as the control vector."""
    well_grid = np.linspace(
        control_parameters.loc_lower,
        control_parameters.loc_upper,
        control_parameters.n_wells_per_side,
    )
    centers = []
    for i in range(control_parameters.n_wells_per_side):
        for j in range(control_parameters.n_wells_per_side):
            centers.append((well_grid[i], well_grid[j]))
    return np.asarray(centers, dtype=float)



def build_control_mollifiers(state_space, control_parameters: ControlParameters):
    """Build the Gaussian well basis used to map coefficients to a source field."""
    centers = control_well_centers(control_parameters)
    width = float(control_parameters.well_width)
    amplitude = 1.0 / (width * math.sqrt(2.0 * math.pi))
    mollifier_list = []
    for center_x, center_y in centers:
        expr = dl.Expression(
            "a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(2*b*b))",
            xi=float(center_x),
            yj=float(center_y),
            a=amplitude,
            b=width,
            mpi_comm=state_space.mesh().mpi_comm(),
            degree=2,
        )
        mollifier_list.append(dl.interpolate(expr, state_space))
    return centers, mollifier_list



def control_coefficients_to_function(function_space, control_vector_or_array, control_parameters: ControlParameters):
    """Convert Gaussian-well coefficients into the induced scalar source field."""
    if hasattr(control_vector_or_array, 'get_local'):
        coeffs = np.array(control_vector_or_array.get_local(), copy=True)
    else:
        coeffs = np.array(control_vector_or_array, dtype=float, copy=True)

    _, mollifier_list = build_control_mollifiers(function_space, control_parameters)
    if coeffs.size != len(mollifier_list):
        raise ValueError(
            f'Expected {len(mollifier_list)} control coefficients, received {coeffs.size}.'
        )

    control_fun = dl.Function(function_space)
    control_fun.vector().zero()
    for coeff, mollifier in zip(coeffs, mollifier_list):
        control_fun.vector().axpy(float(coeff), mollifier.vector())
    control_fun.vector().apply("")
    return control_fun



def make_wave_control_coefficients(control_parameters: ControlParameters, amplitude: float) -> np.ndarray:
    """A deterministic nonzero coefficient pattern for regression tests."""
    centers = control_well_centers(control_parameters)
    coeffs = amplitude * np.sin(np.pi * centers[:, 0]) * np.sin(np.pi * centers[:, 1])
    return np.asarray(coeffs, dtype=float)


class ControlledSemilinearADRWellVarfHandler:
    """Add the Gaussian-well coefficient control as the entire PDE source."""

    def __init__(self, Vh, pde_parameters, control_parameters: ControlParameters):
        self.base_varf_handler = SemilinearEllipticVarfHandler(Vh, pde_parameters)
        _, self.mollifier_list = build_control_mollifiers(Vh[hp.STATE], control_parameters)
        self.mollifiers = dl.as_vector(self.mollifier_list)
        if Vh[soupy.CONTROL].dim() != len(self.mollifier_list):
            raise ValueError(
                'Control dimension and number of Gaussian wells do not match: '
                f"{Vh[soupy.CONTROL].dim()} != {len(self.mollifier_list)}"
            )

    def __call__(self, u, m, p, z):
        return self.base_varf_handler(u, m, p) - dl.inner(self.mollifiers, z) * p * dl.dx



def setup_pde(Vh, pde_parameters):
    """Set up the semilinear ADR PDE with homogeneous Dirichlet data on all boundaries."""
    bc = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), "on_boundary")
    bc0 = dl.DirichletBC(Vh[hp.STATE], dl.Constant(0.0), "on_boundary")
    pde_varf = SemilinearEllipticVarfHandler(Vh, pde_parameters)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)
    return pde



def setup_prior(Vh, prior_parameters):
    """Setup the prior from the function space and prior parameters."""
    m_mean = dl.interpolate(dl.Constant(prior_parameters.mean), Vh[hp.PARAMETER])
    prior = hp.BiLaplacianPrior(
        Vh[hp.PARAMETER],
        prior_parameters.gamma,
        prior_parameters.delta,
        mean=m_mean.vector(),
        robin_bc=prior_parameters.robin_bc,
    )
    return prior


class SemilinearEllipticVarfHandler:
    """Variational form for the semilinear elliptic PDE."""

    def __init__(self, Vh, parameters):
        self.Vh = Vh
        self.parameters = parameters
        self.reaction = dl.Constant(self.parameters.reaction)

    def __call__(self, u, m, p):
        return (
            dl.exp(m) * dl.inner(dl.grad(u), dl.grad(p)) * dl.dx
            + self.reaction * u**3 * p * dl.dx
        )



def _l2_norm(u, m):
    del m
    return u**2 * dl.dx



def _energy_norm(u, m):
    return dl.exp(m) * dl.inner(dl.grad(u), dl.grad(u)) * dl.dx



def _cubic(u, m):
    del m
    return u**3 * dl.dx


class MismatchQoIVarf:
    """Whole-domain mismatch QoI against u_tar = sin(2pi x) sin(2pi y)."""

    def __init__(self, mesh):
        self.target = dl.Expression(
            "sin(2*pi*x[0])*sin(2*pi*x[1])",
            pi=np.pi,
            degree=4,
            mpi_comm=mesh.mpi_comm(),
        )

    def __call__(self, u, m):
        del m
        return (u - self.target) ** 2 * dl.dx



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
        return -dl.inner(dl.exp(m) * dl.grad(u), self.n) * self.ds(1)



def setup_qoi(Vh, qoi_type, mesh):
    """Setup the quantity of interest from the function space and observation parameters."""
    if qoi_type == "mismatch":
        qoi_varf = MismatchQoIVarf(mesh)
    elif qoi_type == "l2":
        qoi_varf = _l2_norm
    elif qoi_type == "energy":
        qoi_varf = _energy_norm
    elif qoi_type == "cubic":
        qoi_varf = _cubic
    elif qoi_type == 'flux':
        qoi_varf = BoundaryFluxVarf(mesh)
    else:
        raise ValueError("Unsupported qoi_type")
    return hp.VariationalQoi(Vh, qoi_varf)
