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
    """
    Base class for a constraint for which a projection operator \
        into the feasible set is available
    """
    def project(self, z):
        """
        Projects the vector :code:`z` onto the constraint set

        :param z: 
        :type z: :py:class:`dolfin.Vector`
        """
        raise NotImplementedError("Child class should implement project")

    def cost(self, z):
        """
        Returns the amount of violation of the constraint by :code:`z`

        :param z: 
        :type z: :py:class:`dolfin.Vector`
        """
        raise NotImplementedError("Child class should implement cost")


class InnerProductEqualityConstraint(ProjectableConstraint):
    """
    Class implements the constraint :math:`c^T z - a = 0`
    """
    def __init__(self, c, a):
        """
        :param c: Constraint vector 
        :type c: :py:class:`dolfin.Vector`
        :param a: Value of the constraint 
        :type a: float 
        """
        self.c = c
        self.a = a 
        self.c_norm2 = self.c.inner(self.c)


    def project(self, z):
        """
        Projects the vector :code:`z` onto the hyperplane \
            :math:`\{z : c^T z = a\}`

        :param z: 
        :type z: :py:class:`dolfin.Vector`
        """
        factor = (self.c.inner(z) - self.a)/self.c_norm2
        z.axpy(-factor, self.c)

    def cost(self, z):
        """
        Returns constraint violation math:`c^T z - a` for input :code:`z`
        :param z: 
        :type z: :py:class:`dolfin.Vector`

        :return: :math:`c^T z - a` for input :code:`z`
        """
        return self.c.inner(z) - self.a

