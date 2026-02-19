"""
Generate a cloaking mesh with a circular hole and cloak ring using gmsh.

Creates physical tags for:
  0: PML region (outer square ring)
  1: Host region (between cloak and PML)
  2: Cloak region (annulus)
Boundary tags:
  1: Outer boundary (Dirichlet)
  2: Inner hole boundary (sound-hard Neumann)

Outputs:
  helmholtz_cloak.msh
  helmholtz_cloak.xdmf (cells)
  helmholtz_cloak_facets.xdmf (facets)
"""

from __future__ import annotations

import os
from pathlib import Path

import gmsh
import meshio


def _gmsh_tag_list(entities):
    return [tag for dim, tag in entities]


def generate_mesh(out_dir: Path, nx: int = 64):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Geometry parameters (paper-scale)
    domain_half = 6.0
    pml_thickness = 1.0
    r_obstacle = 1.0
    r_cloak = 3.0

    gmsh.initialize()
    gmsh.model.add("helmholtz_cloak")

    # Base shapes
    outer = gmsh.model.occ.addRectangle(
        -domain_half, -domain_half, 0.0, 2 * domain_half, 2 * domain_half
    )
    inner = gmsh.model.occ.addRectangle(
        -(domain_half - pml_thickness),
        -(domain_half - pml_thickness),
        0.0,
        2 * (domain_half - pml_thickness),
        2 * (domain_half - pml_thickness),
    )
    circle_outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, r_cloak, r_cloak)
    circle_inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, r_obstacle, r_obstacle)

    # Regions via boolean ops
    cloak, _ = gmsh.model.occ.cut(
        [(2, circle_outer)], [(2, circle_inner)], removeObject=False, removeTool=False
    )
    host, _ = gmsh.model.occ.cut(
        [(2, inner)], [(2, circle_outer)], removeObject=False, removeTool=False
    )
    pml, _ = gmsh.model.occ.cut(
        [(2, outer)], [(2, inner)], removeObject=False, removeTool=False
    )

    gmsh.model.occ.synchronize()

    # Physical groups for subdomains
    gmsh.model.addPhysicalGroup(2, _gmsh_tag_list(pml), 0)
    gmsh.model.addPhysicalGroup(2, _gmsh_tag_list(host), 1)
    gmsh.model.addPhysicalGroup(2, _gmsh_tag_list(cloak), 2)

    gmsh.model.setPhysicalName(2, 0, "pml")
    gmsh.model.setPhysicalName(2, 1, "host")
    gmsh.model.setPhysicalName(2, 2, "cloak")

    # Boundary tags: outer square and inner hole
    outer_curves = _gmsh_tag_list(gmsh.model.getBoundary([(2, outer)], oriented=False))
    inner_curves = _gmsh_tag_list(gmsh.model.getBoundary([(2, circle_inner)], oriented=False))

    gmsh.model.addPhysicalGroup(1, outer_curves, 1)
    gmsh.model.addPhysicalGroup(1, inner_curves, 2)
    gmsh.model.setPhysicalName(1, 1, "outer")
    gmsh.model.setPhysicalName(1, 2, "inner")

    # Mesh sizing (coarse but consistent with nx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2 * domain_half / nx)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2 * domain_half / nx)

    gmsh.model.mesh.generate(2)

    msh_path = out_dir / "helmholtz_cloak.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()

    # Convert to XDMF (cells + facets).
    msh = meshio.read(msh_path)
    tri = msh.cells_dict.get("triangle")
    tri_data = msh.cell_data_dict.get("gmsh:physical", {}).get("triangle")
    line = msh.cells_dict.get("line")
    line_data = msh.cell_data_dict.get("gmsh:physical", {}).get("line")

    if tri is None or tri_data is None:
        raise RuntimeError("Triangle cells or physical tags missing in gmsh mesh.")
    if line is None or line_data is None:
        raise RuntimeError("Line cells or physical tags missing in gmsh mesh.")

    meshio.write(
        out_dir / "helmholtz_cloak.xdmf",
        meshio.Mesh(points=msh.points, cells={"triangle": tri},
                    cell_data={"name_to_read": [tri_data]}),
    )
    meshio.write(
        out_dir / "helmholtz_cloak_facets.xdmf",
        meshio.Mesh(points=msh.points, cells={"line": line},
                    cell_data={"name_to_read": [line_data]}),
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    generate_mesh(here / "mesh", nx=64)
    print("Wrote mesh to", here / "mesh")
