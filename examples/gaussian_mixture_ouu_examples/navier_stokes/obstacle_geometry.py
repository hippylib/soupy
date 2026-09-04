import dolfin as dl 
import numpy as np 

# For labelling 
LEFT = 1 
TOP = 2
RIGHT = 3
BOTTOM = 4
OBSTACLE = 5 

class ObstacleGeometry:
    def __init__(self, mesh, Lx, Ly, body_box):
        self.mesh = mesh
        self.Lx = Lx 
        self.Ly = Ly 
        self.min_x, self.min_y, self.max_x, self.max_y = body_box
        self.body_box = body_box 
        self.LEFT = LEFT
        self.TOP = TOP
        self.RIGHT = RIGHT
        self.BOTTOM = BOTTOM
        self.OBSTACLE = OBSTACLE 
        self.ds = createLabels(mesh, Lx, Ly, body_box)

class ObstacleSubdomain(dl.SubDomain):
    """
    Mark everything within the box [min_x, max_x] \times [min_y, \max_y]
    as being the obstacle 
    """
    def __init__(self, min_x, min_y, max_x, max_y, **kwargs):
        super().__init__(**kwargs)
        self.min_x = min_x 
        self.min_y = min_y
        self.max_x = max_x 
        self.max_y = max_y 

    def inside(self, x, on_boundary):
        return on_boundary and x[0] > self.min_x and x[0] < self.max_x and x[1] > self.min_y and x[1] < self.max_y

def createLabels(mesh, Lx, Ly, body_box):
    """
    Mark boundaries of a domain corresponding to a channel domain
    """
    boundaries = dl.MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
    boundaries.set_all(0)

    left = Left(Lx, Ly)
    right = Right(Lx, Ly)
    top = Top(Lx, Ly)
    bottom = Bottom(Lx, Ly)

    min_x = body_box[0]
    min_y = body_box[1] 
    max_x = body_box[2]
    max_y = body_box[3]

    obstacle = ObstacleSubdomain(min_x, min_y, max_x, max_y)

    left.mark(boundaries, LEFT)
    top.mark(boundaries, TOP)
    right.mark(boundaries, RIGHT)
    bottom.mark(boundaries, BOTTOM)
    obstacle.mark(boundaries, OBSTACLE)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundaries)
    return ds 

class Left(dl.SubDomain):
    def __init__(self, Lx, Ly, **kwargs):
        self.Lx = Lx 
        self.Ly = Ly 
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return dl.near(x[0], 0)

class Top(dl.SubDomain):
    def __init__(self, Lx, Ly, **kwargs):
        self.Lx = Lx 
        self.Ly = Ly 
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return dl.near(x[1], self.Ly)

class Right(dl.SubDomain):
    def __init__(self, Lx, Ly, **kwargs):
        self.Lx = Lx 
        self.Ly = Ly 
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return dl.near(x[0], self.Lx)

class Bottom(dl.SubDomain):
    def __init__(self, Lx, Ly, **kwargs):
        self.Lx = Lx 
        self.Ly = Ly 
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        return dl.near(x[1], 0)