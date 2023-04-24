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
# Software Foundation) version 3.0 dated June 1991.

class ProjectableConstraint:
    def project(self, z):
        raise NotImplementedError("Child class should implement project")

    def cost(self, z):
        raise NotImplementedError("Child class should implement cost")


class InnerProductEqualityConstraint(ProjectableConstraint):
    """
    Class implements the constraint c^T x - a = 0
    """
    def __init__(self, c, a):
        """
        - :code: `c` is constraint `dolfin.Vector()`
        - :code: `a` value of the constraint 
        """
        self.c = c
        self.a = a 
        self.c_norm2 = self.c.inner(self.c)


    def project(self, z):
        factor = (self.c.inner(z) - self.a)/self.c_norm2
        z.axpy(-factor, self.c)

    def cost(self, z):
        """
        returns the amount of violation of the constraint 
        """
        return self.c.inner(z) - self.a
