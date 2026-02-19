# Copyright (c) 2023-2024, The University of Texas at Austin
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

"""
Helmholtz-like cloaking problem setup.

This module sets up an acoustic cloaking problem for optimal design
under uncertainty. The goal is to minimize the scattered field in an observation
region by designing the material properties in a cloak region.

The model equation follows the Taylor-cloak paper (scattered field form) with
the PML split into real/imaginary components:
    Δu + k(x)^2 u = (k0^2 - k(x)^2) u_inc    in Ω
    ∂u/∂n = -∂u_inc/∂n                        on ∂D_o
    u = 0                                     on ∂Ω (truncated/PML boundary)

where:
    - c(x) = exp(m - z) in the cloak region, 1 elsewhere
    - k(x) = k0 / c(x) in the cloak region, k0 elsewhere
    - m is the uncertain parameter (Gaussian random field)
    - z is the design variable (control field in cloak region)
    - u_inc is the incident wave (plane wave)
    - k0 is the background wavenumber

The QoI is the L2 norm of the state in an observation region (host medium),
representing the scattered field that we want to minimize.

Reference:
    Chen, Haberman, Ghattas (2021)
    "Optimal design of acoustic metamaterial cloaks under uncertainty"
    Journal of Computational Physics
"""

import numpy as np
import dolfin as dl
import ufl
from mpi4py import MPI

import sys
import os

# Add paths
_soupy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.environ.get("HIPPYLIB_PATH", ""))
sys.path.append(_soupy_root)

import hippylib as hp

dl.set_log_active(False)

from soupy import (
    ControlModel,
    PDEVariationalControlProblem,
    L2Penalization,
    STATE, PARAMETER, ADJOINT, CONTROL,
)
from soupy.modeling.controlQoI import ControlQoI


class RegionalL2QoI(ControlQoI):
    """QoI: L2 norm of state in a marked region.

    Q(u) = ∫_{Ω_obs} (u1² + u2²) dx for split real/imag state.
    """

    def __init__(self, Vh, region_marker, region_id=1):
        """
        Args:
            Vh: Function space list [STATE, PARAMETER, ADJOINT, CONTROL]
            region_marker: MeshFunction marking subdomains
            region_id: ID of the observation region in the marker
        """
        self.Vh = Vh
        self.dx = dl.Measure('dx', domain=Vh[STATE].mesh(),
                             subdomain_data=region_marker)
        self.region_id = region_id

        # Mass matrix for the observation region (handles mixed or scalar).
        u_trial = dl.TrialFunction(Vh[STATE])
        u_test = dl.TestFunction(Vh[STATE])
        self.M_obs = dl.assemble(dl.inner(u_trial, u_test) * self.dx(region_id))

    def cost(self, x, out=None):
        """Evaluate Q(u) = ∫ u² dx in observation region."""
        u = x[STATE]
        Mu = dl.Vector(u)
        self.M_obs.mult(u, Mu)
        return 0.5 * u.inner(Mu)

    def grad(self, i, x, out):
        """Gradient of Q with respect to component i.

        ∂Q/∂u = u (in observation region, via mass matrix)
        Other gradients are zero.
        """
        out.zero()
        if i == STATE:
            self.M_obs.mult(x[STATE], out)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Set linearization point (for Hessian computations)."""
        pass  # QoI is quadratic, no nonlinear terms

    def adj_rhs(self, x, rhs):
        """Compute the adjoint right-hand side: -∂Q/∂u.

        For Q = 0.5∫u²dx, we have ∂Q/∂u = M_obs * u, so adj_rhs = -M_obs * u.
        """
        rhs.zero()
        self.M_obs.mult(x[STATE], rhs)
        rhs *= -1.0

    def apply_ij(self, i, j, dir, out):
        """Apply Hessian block (i,j) to direction dir.

        For Q = 0.5 u^T M u, we have H_uu = M, others are zero.
        """
        out.zero()
        if i == STATE and j == STATE:
            self.M_obs.mult(dir, out)


class CloakingPDEVarf:
    """Variational form handler for the split PML Helmholtz system.

    Uses the real/imaginary split with PML coefficients (a1..a4, b1, b2).
    """

    def __init__(self, Vh, cloak_marker, cloak_id=2, k0=2*np.pi,
                 incident_dir=(1.0, 0.0), facet_markers=None, inner_id=2,
                 has_obstacle_region=False, domain_half=6.0,
                 pml_thickness=1.0, pml_sigma=25.0):
        """
        Args:
            Vh: Function space list
            cloak_marker: MeshFunction marking cloak region
            cloak_id: ID of cloak region in the marker
            k0: Background wavenumber
            incident_dir: Incident wave direction
            facet_markers: Facet markers for boundary integrals (inner hole)
            inner_id: Marker id for inner hole boundary
            has_obstacle_region: Whether marker=3 exists (fallback mesh without hole)
            domain_half: Half-width of the computational square
            pml_thickness: Thickness of the PML layer
            pml_sigma: Maximum PML damping strength
        """
        self.Vh = Vh
        self.cloak_marker = cloak_marker
        self.cloak_id = cloak_id
        self.k0 = float(k0)
        self.incident_dir = np.array(incident_dir, dtype=float)
        self.facet_markers = facet_markers
        self.inner_id = inner_id
        self.has_obstacle_region = has_obstacle_region
        self.domain_half = float(domain_half)
        self.pml_thickness = float(pml_thickness)
        self.pml_sigma = float(pml_sigma)

        # Measures
        self.dx = dl.Measure('dx', domain=Vh[STATE].mesh(),
                             subdomain_data=cloak_marker)
        if facet_markers is not None:
            self.ds = dl.Measure('ds', domain=Vh[STATE].mesh(),
                                 subdomain_data=facet_markers)
        else:
            self.ds = dl.Measure('ds', domain=Vh[STATE].mesh())

        # Incident plane wave (real/imag parts).
        mesh = Vh[STATE].mesh()
        x = dl.SpatialCoordinate(mesh)
        inc_norm = np.linalg.norm(self.incident_dir)
        if inc_norm == 0.0:
            self.incident_dir = np.array([1.0, 0.0])
        else:
            self.incident_dir = self.incident_dir / inc_norm
        phase = self.k0 * (self.incident_dir[0] * x[0] +
                           self.incident_dir[1] * x[1])
        self.u_inc1 = dl.cos(phase)
        self.u_inc2 = dl.sin(phase)

        # PML damping profiles (quadratic ramp).
        if self.pml_thickness <= 0.0:
            self.sigma_x1 = dl.Constant(0.0)
            self.sigma_x2 = dl.Constant(0.0)
        else:
            abs_x1 = ufl.sqrt(x[0] * x[0])
            abs_x2 = ufl.sqrt(x[1] * x[1])
            ramp_x1 = ufl.max_value(0.0, abs_x1 - (self.domain_half - self.pml_thickness))
            ramp_x2 = ufl.max_value(0.0, abs_x2 - (self.domain_half - self.pml_thickness))
            self.sigma_x1 = self.pml_sigma * (ramp_x1 / self.pml_thickness) ** 2
            self.sigma_x2 = self.pml_sigma * (ramp_x2 / self.pml_thickness) ** 2

    def __call__(self, u, m, p, z):
        """Return the weak form residual.

        Bilinear: Split PML form (Eq. 5.20-5.26 in Taylor-cloak).
        Linear:   Incident-wave forcing and sound-hard obstacle term.
        """
        u1, u2 = dl.split(u)
        v1, v2 = dl.split(p)

        # Sound speed c = exp(m - z) in cloak, 1 elsewhere.
        # k(x) = k0 / c(x) => k^2 = k0^2 * exp(2*(z - m)) in cloak.
        k2_cloak = (self.k0 ** 2) * dl.exp(2.0 * (z - m))
        k2_host = self.k0 ** 2

        # Standard Helmholtz form (sigma = 0) in host + cloak.
        grad_u1 = dl.grad(u1)
        grad_u2 = dl.grad(u2)
        grad_v1 = dl.grad(v1)
        grad_v2 = dl.grad(v2)

        bilinear_host = (
            dl.inner(grad_u1, grad_v1) + dl.inner(grad_u2, grad_v2)
            - k2_host * (u1 * v1 + u2 * v2)
        ) * self.dx(1)

        bilinear_cloak = (
            dl.inner(grad_u1, grad_v1) + dl.inner(grad_u2, grad_v2)
            - k2_cloak * (u1 * v1 + u2 * v2)
        ) * self.dx(self.cloak_id)

        # PML coefficients (k = k0 in PML layer).
        k = self.k0
        sigma_x1 = self.sigma_x1
        sigma_x2 = self.sigma_x2
        a1 = (k**2 + sigma_x1 * sigma_x2) / (k**2 + sigma_x1**2)
        a2 = (k * (sigma_x1 - sigma_x2)) / (k**2 + sigma_x1**2)
        a3 = (k**2 + sigma_x1 * sigma_x2) / (k**2 + sigma_x2**2)
        a4 = (k * (sigma_x2 - sigma_x1)) / (k**2 + sigma_x2**2)
        b1 = k**2 - sigma_x1 * sigma_x2
        b2 = -k * (sigma_x1 + sigma_x2)

        du1_dx1 = ufl.grad(u1)[0]
        du1_dx2 = ufl.grad(u1)[1]
        du2_dx1 = ufl.grad(u2)[0]
        du2_dx2 = ufl.grad(u2)[1]
        dv1_dx1 = ufl.grad(v1)[0]
        dv1_dx2 = ufl.grad(v1)[1]
        dv2_dx1 = ufl.grad(v2)[0]
        dv2_dx2 = ufl.grad(v2)[1]

        bilinear_pml = (
            (a1 * du1_dx1 - a2 * du2_dx1) * dv1_dx1
            + (a3 * du1_dx2 - a4 * du2_dx2) * dv1_dx2
            + (a1 * du2_dx1 + a2 * du1_dx1) * dv2_dx1
            + (a3 * du2_dx2 + a4 * du1_dx2) * dv2_dx2
            - ((b1 * u1 - b2 * u2) * v1 + (b1 * u2 + b2 * u1) * v2)
        ) * self.dx(0)

        # Source term from incident wave (only nonzero where k != k0).
        source_term = (
            (k2_host - k2_cloak) * self.u_inc1 * v1
            + (k2_host - k2_cloak) * self.u_inc2 * v2
        ) * self.dx(self.cloak_id)

        # Sound-hard boundary condition on the hole (scattered field form).
        if not self.has_obstacle_region:
            n = dl.FacetNormal(self.Vh[STATE].mesh())
            dudn_inc1 = dl.dot(dl.grad(self.u_inc1), n)
            dudn_inc2 = dl.dot(dl.grad(self.u_inc2), n)
            source_term += -(dudn_inc1 * v1 + dudn_inc2 * v2) * self.ds(self.inner_id)

        result = bilinear_cloak + bilinear_host + bilinear_pml - source_term

        # Obstacle region (marker = 3) - only for fallback mesh without hole
        if self.has_obstacle_region:
            bilinear_obstacle = (
                100.0 * dl.inner(grad_u1, grad_v1)
                + 100.0 * dl.inner(grad_u2, grad_v2)
                - 100.0 * k2_host * (u1 * v1 + u2 * v2)
            ) * self.dx(3)
            result = result + bilinear_obstacle

        return result


def _load_gmsh_mesh(mesh_dir, comm):
    # Prefer XDMF if present.
    mesh_path = os.path.join(mesh_dir, "helmholtz_cloak.xdmf")
    facet_path = os.path.join(mesh_dir, "helmholtz_cloak_facets.xdmf")

    if os.path.exists(mesh_path) and os.path.exists(facet_path):
        mesh = dl.Mesh()
        with dl.XDMFFile(comm, mesh_path) as xdmf:
            xdmf.read(mesh)

        mvc_cells = dl.MeshValueCollection("size_t", mesh, mesh.topology().dim())
        with dl.XDMFFile(comm, mesh_path) as xdmf:
            xdmf.read(mvc_cells, "name_to_read")
        subdomains = dl.MeshFunction("size_t", mesh, mvc_cells)

        mvc_facets = dl.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
        with dl.XDMFFile(comm, facet_path) as xdmf:
            xdmf.read(mvc_facets, "name_to_read")
        facet_markers = dl.MeshFunction("size_t", mesh, mvc_facets)

        geom = {
            "domain_half": 6.0,
            "pml_thickness": 1.0,
            "center": np.array([0.0, 0.0]),
            "r_obstacle": 1.0,
            "r_cloak": 3.0,
            "facet_markers": facet_markers,
        }

        return mesh, subdomains, geom

    # Fallback to DOLFIN XML files (no h5py dependency).
    mesh_path = os.path.join(mesh_dir, "helmholtz_cloak.xml")
    mesh_markers = os.path.join(mesh_dir, "helmholtz_cloak_name_to_read.xml")
    facet_mesh = os.path.join(mesh_dir, "helmholtz_cloak_facets.xml")
    facet_markers_path = os.path.join(mesh_dir, "helmholtz_cloak_facets_name_to_read.xml")

    if not (os.path.exists(mesh_path) and os.path.exists(mesh_markers) and
            os.path.exists(facet_mesh) and os.path.exists(facet_markers_path)):
        return None

    mesh = dl.Mesh(mesh_path)
    subdomains = dl.MeshFunction("size_t", mesh, mesh_markers)
    facet_markers = dl.MeshFunction("size_t", mesh, facet_markers_path)

    geom = {
        "domain_half": 6.0,
        "pml_thickness": 1.0,
        "center": np.array([0.0, 0.0]),
        "r_obstacle": 1.0,
        "r_cloak": 3.0,
        "facet_markers": facet_markers,
    }

    return mesh, subdomains, geom


def create_cloaking_mesh(nx=32, comm=None):
    """Create mesh with marked regions for cloaking problem.

    This follows the geometry in the Taylor-cloak paper:
        - Outer square domain: [-L, L]^2 with PML thickness t_pml
        - Circular obstacle of radius r_obstacle (removed from the mesh)
        - Cloak annulus between r_obstacle and r_cloak

    Regions (cell markers):
        0: PML/boundary layer
        1: Host medium (observation region)
        2: Cloak region (design region)

    Returns:
        mesh, subdomains (MeshFunction), geom (dict)
    """
    if comm is None:
        comm = MPI.COMM_SELF

    # Try loading a gmsh mesh (preferred, exact circle).
    mesh_dir = os.path.join(os.path.dirname(__file__), "mesh")
    gmsh_loaded = _load_gmsh_mesh(mesh_dir, comm)
    if gmsh_loaded is not None:
        return gmsh_loaded

    # Geometry parameters (paper-scale)
    domain_half = 6.0
    pml_thickness = 1.0
    center = np.array([0.0, 0.0])
    r_obstacle = 1.0
    r_cloak = 3.0

    geom = {
        "domain_half": domain_half,
        "pml_thickness": pml_thickness,
        "center": center,
        "r_obstacle": r_obstacle,
        "r_cloak": r_cloak,
    }

    # Build mesh with a circular hole using mshr (FEniCS 2019.1).
    try:
        import mshr
    except ImportError:
        mesh, subdomains, geom = _create_simple_cloaking_mesh(nx, comm, geom)
        return mesh, subdomains, geom

    square = mshr.Rectangle(
        dl.Point(-domain_half, -domain_half),
        dl.Point(domain_half, domain_half),
    )
    obstacle = mshr.Circle(dl.Point(center[0], center[1]), r_obstacle, 64)
    domain = square - obstacle

    mesh = mshr.generate_mesh(domain, nx)

    # Mark subdomains using MeshFunction (cell markers).
    subdomains = dl.MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomains.set_all(1)  # Host medium by default

    class CloakRegion(dl.SubDomain):
        def inside(self, x, on_boundary):
            r = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
            return (r <= r_cloak + 1e-12) and (r >= r_obstacle - 1e-12)

    class PMLRegion(dl.SubDomain):
        def inside(self, x, on_boundary):
            return (
                abs(x[0]) > domain_half - pml_thickness - 1e-12 or
                abs(x[1]) > domain_half - pml_thickness - 1e-12
            )

    CloakRegion().mark(subdomains, 2)
    PMLRegion().mark(subdomains, 0)

    return mesh, subdomains, geom


def _create_simple_cloaking_mesh(nx=32, comm=None, geom=None):
    """Fallback: Create mesh with a hole using SubMesh (no mshr).

    This removes cells inside the obstacle radius by creating a SubMesh.
    It preserves a true inner boundary so the sound-hard condition is natural.
    """
    if comm is None:
        comm = MPI.COMM_SELF

    if geom is None:
        geom = {
            "domain_half": 6.0,
            "pml_thickness": 1.0,
            "center": np.array([0.0, 0.0]),
            "r_obstacle": 1.0,
            "r_cloak": 3.0,
        }

    domain_half = geom["domain_half"]
    pml_thickness = geom["pml_thickness"]
    center = geom["center"]
    r_obstacle = geom["r_obstacle"]
    r_cloak = geom["r_cloak"]

    full_mesh = dl.RectangleMesh(
        comm,
        dl.Point(-domain_half, -domain_half),
        dl.Point(domain_half, domain_half),
        nx,
        nx,
    )

    # Mark cells to keep (exclude obstacle)
    cell_markers = dl.MeshFunction("size_t", full_mesh, full_mesh.topology().dim(), 1)
    for cell in dl.cells(full_mesh):
        mp = cell.midpoint()
        r = np.sqrt((mp.x() - center[0])**2 + (mp.y() - center[1])**2)
        if r < r_obstacle:
            cell_markers[cell] = 0
        else:
            cell_markers[cell] = 1

    mesh = dl.SubMesh(full_mesh, cell_markers, 1)

    # Create subdomain markers on the submesh
    subdomains = dl.MeshFunction("size_t", mesh, mesh.topology().dim(), 1)
    for cell in dl.cells(mesh):
        mp = cell.midpoint()
        r = np.sqrt((mp.x() - center[0])**2 + (mp.y() - center[1])**2)

        if r < r_cloak:
            subdomains[cell] = 2  # Cloak region
        elif (abs(mp.x()) > domain_half - pml_thickness or
              abs(mp.y()) > domain_half - pml_thickness):
            subdomains[cell] = 0  # PML/boundary layer
        else:
            subdomains[cell] = 1  # Host medium (observation)

    return mesh, subdomains, geom


def setup_helmholtz_cloaking_problem(nx=32, wavenumber=2*np.pi,
                                      prior_gamma=10.0, prior_delta=50.0,
                                      penalty_alpha=1e-3, incident_dir=(1.0, 0.0),
                                      comm=None):
    """Set up the Helmholtz cloaking problem.

    Args:
        nx: Mesh resolution
        wavenumber: k0 background wavenumber
        prior_gamma: Prior regularization (BiLaplacian)
        prior_delta: Prior regularization (BiLaplacian)
        penalty_alpha: L2 penalty weight on control
        incident_dir: Incident wave direction (unit vector)
        comm: MPI communicator

    Returns:
        mesh, Vh, control_model, prior, penalty, problem_info
    """
    if comm is None:
        comm = MPI.COMM_SELF

    # Create mesh with region markers
    mesh, subdomains, geom = create_cloaking_mesh(nx, comm)

    # Function spaces (split real/imag state)
    Vh_STATE = dl.VectorFunctionSpace(mesh, "CG", 1, dim=2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh_CONTROL = dl.FunctionSpace(mesh, "CG", 1)

    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

    # Check if mesh has obstacle region (fallback mesh without hole).
    has_obstacle_region = 3 in subdomains.array()

    # Boundary conditions:
    # - Outer boundary: Dirichlet u=0 (absorbing-like)
    # - Inner boundary (obstacle): sound-hard scattered-field BC
    domain_half = geom["domain_half"]
    center = geom["center"]
    r_obstacle = geom["r_obstacle"]

    facet_markers = geom.get("facet_markers")
    if facet_markers is None:
        facet_markers = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

        class OuterBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                if not on_boundary:
                    return False
                tol = 1e-10
                return (abs(x[0]) > domain_half - tol or
                        abs(x[1]) > domain_half - tol)

        class InnerBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                if not on_boundary:
                    return False
                r = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
                return abs(r - r_obstacle) < 1e-6

        OuterBoundary().mark(facet_markers, 1)

        # Only mark the inner boundary when the obstacle is an actual hole.
        if not has_obstacle_region:
            InnerBoundary().mark(facet_markers, 2)

    bc = dl.DirichletBC(Vh_STATE, dl.Constant((0.0, 0.0)), facet_markers, 1)
    bc0 = dl.DirichletBC(Vh_STATE, dl.Constant((0.0, 0.0)), facet_markers, 1)

    # PDE variational form
    pde_varf = CloakingPDEVarf(
        Vh,
        subdomains,
        cloak_id=2,
        k0=wavenumber,
        incident_dir=incident_dir,
        facet_markers=facet_markers,
        inner_id=2,
        has_obstacle_region=has_obstacle_region,
        domain_half=geom["domain_half"],
        pml_thickness=geom["pml_thickness"],
    )

    # Create PDE problem with direct solver for stability
    pde = PDEVariationalControlProblem(
        Vh, pde_varf, [bc], [bc0], is_fwd_linear=True, lu_method="default"
    )

    # Prior for uncertain parameter
    mean_vector = dl.interpolate(dl.Constant(0.0), Vh_PARAMETER).vector()
    prior = hp.BiLaplacianPrior(
        Vh_PARAMETER, prior_gamma, prior_delta, mean=mean_vector, robin_bc=True
    )

    # QoI: scattered field in observation region (region 1)
    qoi = RegionalL2QoI(Vh, subdomains, region_id=1)

    # Control model
    control_model = ControlModel(pde, qoi)

    # Penalization
    penalty = L2Penalization(Vh, penalty_alpha)

    # Problem info dict
    problem_info = {
        'subdomains': subdomains,
        'wavenumber': wavenumber,
        'center': geom["center"],
        'r_obstacle': geom["r_obstacle"],
        'r_cloak': geom["r_cloak"],
        'pml_thickness': geom["pml_thickness"],
        'domain_half': geom["domain_half"],
        'has_hole': not has_obstacle_region,
        'facet_markers': facet_markers,
        'incident_dir': incident_dir,
    }

    return mesh, Vh, control_model, prior, penalty, problem_info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Setting up Helmholtz cloaking problem...")

    mesh, Vh, control_model, prior, penalty, info = setup_helmholtz_cloaking_problem(
        nx=32
    )

    print(f"Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")
    print(f"State DOFs: {Vh[STATE].dim()}")
    print(f"Parameter DOFs: {Vh[PARAMETER].dim()}")
    print(f"Control DOFs: {Vh[CONTROL].dim()}")

    # Plot subdomains
    plt.figure(figsize=(8, 8))
    c = dl.plot(info['subdomains'])
    plt.colorbar(c)
    plt.title("Domain regions\n(0=PML, 1=Host, 2=Cloak, 3=Obstacle)")
    plt.savefig("helmholtz_domains.png", dpi=150)
    print("Saved helmholtz_domains.png")

    # Test forward solve
    print("\nTesting forward solve at prior mean...")
    x = [dl.Function(Vh[i]).vector() for i in range(4)]
    x[PARAMETER].axpy(1.0, prior.mean)

    try:
        control_model.solveFwd(x[STATE], x)
        print("  Forward solve succeeded!")

        # Evaluate QoI
        qoi_val = control_model.cost(x)
        print(f"  QoI value: {qoi_val:.6e}")

        # Plot solution
        plt.figure(figsize=(8, 8))
        u_func = hp.vector2Function(x[STATE], Vh[STATE])
        u1, u2 = u_func.split(deepcopy=True)
        c = dl.plot(u1)
        plt.colorbar(c)
        plt.title("State solution (real part, prior mean)")
        plt.savefig("helmholtz_state.png", dpi=150)
        print("  Saved helmholtz_state.png")

    except Exception as e:
        print(f"  Forward solve failed: {e}")

    print("\nSetup complete!")
