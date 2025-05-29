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
# Software Foundation) version 3.0 dated June 2007.

import unittest 
import dolfin as dl
import numpy as np

import sys
import os 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')
from soupy import AugmentedVector, set_local_from_global

class TestAugmentedVectorMeshParallel(unittest.TestCase):
    """
    Test cases for :code:`AugmentedVector` in the case where 
    mesh is partitioned in MPI 
    """
    def setUp(self):
        mesh = dl.UnitIntervalMesh(10)
        self.V = dl.FunctionSpace(mesh, "CG", 2)
        self.dim = self.V.dim()
        self.atol = 1e-6

    def _construct_test_vector(self, copy_vector=False):
        z_fun = dl.Function(self.V)
        z_fun.interpolate(dl.Expression("sin(x[0])", degree=4))

        z = z_fun.vector()
        zt = AugmentedVector(z, copy_vector=copy_vector) 
        return z, zt 

    def testConstructor(self):
        """
        Check that each process gets the partitioned vector :code:`z`
        and the full scalar :code:`t`
        """
        z, zt = self._construct_test_vector()
        t_true = 2.0  
        zt.set_scalar(t_true)
        
        z_get = zt.get_vector()
        t_get = zt.get_scalar()


        z_diff = z - z_get 
        z_diff_norm = np.sqrt(z_diff.inner(z_diff))
        self.assertTrue(z_diff_norm <= self.atol)

        self.assertTrue(np.isclose(t_get, t_true))

    def testGetLocal(self):
        """
        Test that :code:`AugmentedVector.get_local()` 
        retrieves a local vector that contains the underlying 
        vector :code:`v.get_local()` concatenated to the scalar :code:`t`
        i.e. it should be :code:`[v_local, t]`
        """
        z, zt = self._construct_test_vector(copy_vector=False)
        t = 2.0
        zt.set_scalar(t)

        z_get = zt.get_vector()

        # Check local objects are correct 
        z_local = z.get_local()
        zt_local = zt.get_local()

        self.assertTrue(len(zt_local) == len(z_local) + 1)
        self.assertTrue(np.allclose(z_local, zt_local[:-1]))
        self.assertTrue(np.isclose(zt_local[-1], t))

    def testInner(self):
        """
        Test the inner product :code:`AugmentedVector.inner(v_other)`
        Uses :code:`set_local_from_global()` which is tested in another 
        test case under :code:`ptest_mpiUtils.py`
        """
        z1 = dl.Function(self.V).vector()
        z2 = dl.Function(self.V).vector()
        
        # Need to fix seed to ensure each process gets the same 
        zt1_np = np.arange(self.dim+1)
        zt2_np = np.arange(self.dim+1) / 2
        zt1 = AugmentedVector(z1)
        zt2 = AugmentedVector(z2)
        set_local_from_global(zt1, zt1_np)
        set_local_from_global(zt2, zt2_np)

        ip = zt1.inner(zt2)
        self.assertTrue(np.allclose([ip], [np.inner(zt1_np, zt2_np)]))


if __name__ == "__main__":
    unittest.main()

