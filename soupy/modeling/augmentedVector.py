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

import dolfin as dl
import numpy as np

class AugmentedVector:
    """
    Class representing an augmented optimization variable (z, t), where t
    is a real number. Methods mirror that of dl.Vector()
    Assumes the last element in the array is the t variable
    """
    def __init__(self, v, copy_vector=True):
        if copy_vector:
            self.v = v.copy()
        else:
            self.v = v
        self.t = 0.0
        self.v_dim = len(self.v.get_local())

    def copy(self):
        x = AugmentedVector(self.v, copy_vector=True)
        x.set_scalar(self.t)


    def add_local(self, vt_array):
        self.v.add_local(vt_array[:-1])
        self.v.apply("")
        self.t += vt_array[-1]

    def set_local(self, vt_array):
        self.v.set_local(vt_array[:-1])
        self.v.apply("")
        self.t = vt_array[-1]

    def apply(self, method):
        self.v.apply(method)

    def get_local(self):
        return np.append(self.v.get_local(), self.t)

    def zero(self):
        self.v.zero()
        self.t = 0.0

    def get_vector(self):
        return self.v

    def set_vector(self, v):
        self.v = v

    def get_scalar(self):
        return self.t

    def set_scalar(self,t):
        self.t = t

    def axpy(self, a, vt):
        self.v.axpy(a, vt.get_vector())
        self.t += a * vt.get_scalar()

    def inner(self, vt):
        value = self.get_vector().inner(vt.get_vector()) + self.get_scalar() * vt.get_scalar() 
        return value 
    
    def apply(self, method):
        self.v.apply(method)

if __name__ == "__main__":
    dim = 5
    mesh = dl.UnitIntervalMesh(10)
    V = dl.VectorFunctionSpace(mesh, "R", 0, dim=dim)
    z1 = dl.Function(V).vector()
    z2 = dl.Function(V).vector()

    zt1_np = np.random.randn(dim + 1)
    zt2_np = np.random.randn(dim + 1)

    zt1 = AugmentedVector(z1)
    zt2 = AugmentedVector(z2)

    print("Initial vectors")
    print("zt1: ", zt1.get_local())
    print("zt2: ", zt2.get_local())

    a = 0.5

    print("Set local")
    zt1.set_local(zt1_np)
    zt2.set_local(zt2_np)
    print("zt1: ", zt1.get_local())
    print("zt2: ", zt2.get_local())

    print("(z1,t): ", zt1.get_vector().get_local(), zt1.get_scalar())
    print("(z1,t): ", zt2.get_vector().get_local(), zt2.get_scalar())

    print("should be")
    print(zt1_np)
    print(zt2_np)


    print("Add local")
    zt1.add_local(zt2.get_local())
    print("zt1: ", zt1.get_local())
    print("should be")
    print(zt1_np + zt2_np)

    print("AXPY")
    zt1.axpy(a, zt2)
    print("zt1: ", zt1.get_local())
    print("should be")
    print(zt1_np + zt2_np + a * zt2_np)


    print("ZERO")
    zt1.zero()
    print("zt1: ", zt1.get_local())
    print("zt2: ", zt2.get_local())




