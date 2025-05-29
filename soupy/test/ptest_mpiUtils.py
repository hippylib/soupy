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
from mpi4py import MPI 

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
sys.path.append('../../')
from soupy import AugmentedVector, set_local_from_global, get_global

class TestParallelGetSet(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.mesh = dl.UnitIntervalMesh(self.comm, 10)
        self.V = dl.FunctionSpace(self.mesh, "CG", 2)
        self.dim = self.V.dim()
        self.atol = 1e-6
        self.comm_size = self.comm.Get_size()
        self.comm_rank = self.comm.Get_rank()

    def _construct_test_vector(self):
        z_fun = dl.Function(self.V)
        z_fun.interpolate(dl.Expression("sin(x[0])", degree=4))
        return z_fun.vector()

    def testVectorGetGlobal(self):
        """
        Check that all processes using get_global get the gathered case 
        Uses numpy i/o to check mpi comm 
        """
        z = self._construct_test_vector()
        z_np = get_global(z)
        
        # Get the gathered form of original vector as ground truth 
        z_gathered_on_zero = z.gather_on_zero()
        if self.comm_rank == 0:
            np.save(f"temp_z.npy", z_gathered_on_zero)
        self.comm.Barrier()

        # Load in ground truth on all of the processes and compare 
        z_gathered_on_zero = np.load(f"temp_z.npy")
        self.assertTrue(np.allclose(z_gathered_on_zero, z_np))

        # Clean up 
        if self.comm_rank == 0:
            os.system("rm temp_z.npy")
        self.comm.Barrier()



    def testVectorSetGlobal(self):
        """
        Check that all processes using get_global get the gathered case 
        Uses numpy i/o to check mpi comm 
        """
        z = self._construct_test_vector()
        z_np = get_global(z)
        
        # Completely zeroed vector
        z_new = dl.Function(self.V).vector()
        set_local_from_global(z_new, z_np)
        
        z_diff = z - z_new 
        z_diff_norm = np.sqrt(z_diff.inner(z_diff))
        self.assertTrue(z_diff_norm <= self.atol)

        


    def testAugmentedVectorGetGlobal(self):
        """
        Check that all processes using set_global properly sets the gathered vector
        """
        z = self._construct_test_vector()
        zt = AugmentedVector(z)
        t = 2.8 
        zt.set_scalar(t)
        
        
        # Get the global from function 
        zt_np = get_global(zt)
        z_np = zt_np[:-1]
        t_np = zt_np[-1]
        
        # Get the gathered form of original vector as ground truth 
        z_gathered_on_zero = z.gather_on_zero()
        if self.comm_rank == 0:
            np.save(f"temp_z.npy", z_gathered_on_zero)
        self.comm.Barrier()
        
        # Load in ground truth on all of the processes
        z_gathered_on_zero = np.load(f"temp_z.npy")

        # Compare the vector components 
        self.assertTrue(np.allclose(z_gathered_on_zero, z_np))

        # Compare the global scalar to true scalar
        self.assertTrue(t == t_np) 

        # Clean up 
        if self.comm_rank == 0:
            os.system("rm temp_z.npy")
        self.comm.Barrier()



    def testAugmentedVectorSetGlobal(self):
        """
        Check that all processes using set_global properly sets the gathered vector
        """
        z = self._construct_test_vector()
        t = -2.8 
        zt = AugmentedVector(z) 
        zt.set_scalar(t)


        zt_np = get_global(zt)
        
        # Completely zeroed vector
        z_new = dl.Function(self.V).vector()
        zt_new = AugmentedVector(z_new)
        set_local_from_global(zt_new, zt_np)
        
        # pull out internals        
        z_new = zt_new.get_vector()
        t_new = zt_new.get_scalar()
        
        # Check vector difference
        z_diff = z - z_new 
        z_diff_norm = np.sqrt(z_diff.inner(z_diff))
        self.assertTrue(z_diff_norm <= self.atol)
        
        # Check scalar difference 
        self.assertTrue(np.abs(t - t_new)< self.atol)


if __name__ == "__main__":
    unittest.main()
