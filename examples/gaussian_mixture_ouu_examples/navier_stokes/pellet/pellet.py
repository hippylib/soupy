import os 
import argparse
import pickle 
import pygmsh 
import gmsh 
import meshio 
import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt 

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh


def rotate_points(p_list, alpha, translate=np.array([0.0, 0.0, 0.0])):
    """ 
    Rotate points clockwise by alpha degrees
    assumes input p_list consists of elements p = [x, y, z]
    rotation leaves z coordinate unchanged
    Adds optional translation
    """
    alpha_rad = alpha/180 * np.pi 
    rotation_matrix = np.array([[np.cos(alpha_rad), np.sin(alpha_rad), 0], [-np.sin(alpha_rad), np.cos(alpha_rad), 0], [0, 0, 1]])
    p_rotated_list = [translate + rotation_matrix @ p for p in p_list]
    return p_rotated_list 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solving OUU problem")
    parser.add_argument('-m', '--mesh_size', choices=["fine", "medium", "medium_coarse", "coarse", "coarsest"], type=str, default="coarse", help="mesh size")
    parser.add_argument('-L', '--L', type=float, default=2.0, help="Length of domain")
    parser.add_argument('-H', '--H', type=float, default=1.0, help="Height of domain")
    parser.add_argument('-R', '--R', type=float, default=0.1, help="Radius of obstacle")
    parser.add_argument('-a', '--angle', type=float, default=0, help="Angle of attack")
    parser.add_argument('-x', '--x', type=float, default=0.5, help="x coordinate of center")
    parser.add_argument('-y', '--y', type=float, default=0.5, help="y coordinate of center")
    parser.add_argument('-c', '--coarsen', type=float, default=2, help="Ratio for coarsening of mesh to the right of domain")
    args = parser.parse_args()

    mesh_sizes = {"fine" : 0.01, "medium" : 0.02, "medium_coarse" : 0.03, "coarse" : 0.04, "coarsest" : 0.05}
    MESH_SIZE = args.mesh_size
    RESOLUTION = mesh_sizes[MESH_SIZE]
    COARSEN_RIGHT_RATIO = args.coarsen
    L = args.L
    H = args.H
    CX = args.x
    CY = args.y
    R = args.R
    ANGLE = args.angle

    mesh_name = "pellet_%s_r%g_angle%g" %(MESH_SIZE, R, ANGLE)
    os.makedirs(mesh_name, exist_ok=True)

    # ---------------------------- #
    # Pre-process the coordinates 
    geometry = dict()
    geometry["lx"] = L
    geometry["ly"] = H 
    geometry["center"] = [CX, CY]
    geometry["radius"] = R 
    geometry["box"] = [CX - 2.5 * R, CY - 2.5 * R, CX + 2.5 * R, CY + 2.5 * R] # Box around obstacle

    center_coord = np.array([CX, CY, 0.0])
    topLeftCoord = np.array([-R, R, 0])
    midLeftCoord = np.array([-R, 0, 0])
    botLeftCoord = np.array([-R, -R, 0])
    topRightCoord = np.array([R, R, 0])
    midRightCoord = np.array([R, 0, 0])
    botRightCoord = np.array([R, -R, 0])

    allBodyCoords = [topLeftCoord, midLeftCoord, botLeftCoord, topRightCoord, midRightCoord, botRightCoord]
    rotatedBodyCoords = rotate_points(allBodyCoords, ANGLE, translate=center_coord)
    topLeftCoord, midLeftCoord, botLeftCoord, topRightCoord, midRightCoord, botRightCoord = rotatedBodyCoords
    
    # Store control edge as x0, x1, y0, y1
    control_edge_top = [topLeftCoord[0], topRightCoord[0], topLeftCoord[1], topRightCoord[1]]
    control_edge_bot = [botLeftCoord[0], botRightCoord[0], botLeftCoord[1], botRightCoord[1]]

    geometry["control_edge_top"] = control_edge_top
    geometry["control_edge_bot"] = control_edge_bot

    with open("%s/geometry.p" %(mesh_name), "wb") as geofile:
        pickle.dump(geometry, geofile)

    # ---------------------------- #
    # Start generating mesh
    
    geometry = pygmsh.geo.Geometry()
    model = geometry.__enter__()
    
    # Domain
    points = [model.add_point((0,0,0), mesh_size=RESOLUTION),
            model.add_point((L,0,0), mesh_size=COARSEN_RIGHT_RATIO*RESOLUTION),
            model.add_point((L,H,0), mesh_size=COARSEN_RIGHT_RATIO*RESOLUTION),
            model.add_point((0,H,0), mesh_size=RESOLUTION)]

    channel_lines = [model.add_line(points[i], points[i+1]) for i in range(-1, len(points)-1)]
    channel_loop = model.add_curve_loop(channel_lines)

    # Obstacle
    topLeft = model.add_point(topLeftCoord, mesh_size=RESOLUTION/2)
    midLeft = model.add_point(midLeftCoord, mesh_size=RESOLUTION/2)
    botLeft = model.add_point(botLeftCoord, mesh_size=RESOLUTION/2)
    topRight = model.add_point(topRightCoord, mesh_size=RESOLUTION/2)
    midRight = model.add_point(midRightCoord, mesh_size=RESOLUTION/2)
    botRight = model.add_point(botRightCoord, mesh_size=RESOLUTION/2)

    bluffbody_lines = [model.add_circle_arc(topLeft, midLeft, botLeft)]
    bluffbody_lines.append(model.add_line(botLeft, botRight))
    bluffbody_lines.append(model.add_line(botRight, topRight))
    bluffbody_lines.append(model.add_line(topRight, topLeft))
    bluffbody_loop = model.add_curve_loop(bluffbody_lines)

    plane_surface = model.add_plane_surface(channel_loop, holes=[bluffbody_loop])
    model.synchronize()
    
    # Mark boundaries
    volume_marker = 6
    model.add_physical([plane_surface], "Volume")
    model.add_physical([channel_lines[0]], "Inflow")
    model.add_physical([channel_lines[2]], "Outflow")
    model.add_physical([channel_lines[1], channel_lines[3]], "Walls")
    model.add_physical(bluffbody_loop.curves, "Obstacle")

    geometry.generate_mesh(dim=2)
    gmsh.write("%s/mesh.msh" %(mesh_name))
    gmsh.clear()
    geometry.__exit__()

    # ------ Convert mesh to xdmf for reading in dolfin ------ # 
    mesh_from_file = meshio.read("%s/mesh.msh" %(mesh_name))
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("%s/facet_mesh.xdmf" %(mesh_name), line_mesh)
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("%s/mesh.xdmf" %(mesh_name), triangle_mesh)

    mesh = dl.Mesh()
    mesh_load = "%s/mesh.xdmf" %(mesh_name)
    f = dl.XDMFFile(dl.MPI.comm_world, mesh_load)
    f.read(mesh)
    dl.plot(mesh)
    plt.show()