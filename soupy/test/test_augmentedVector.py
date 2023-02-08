import unittest 
import dolfin as dl
import numpy as np

import sys
sys.path.append('../../')
from soupy import AugmentedVector

class TestAugmentedVector(unittest.TestCase):
    def setUp(self):
        mesh = dl.UnitIntervalMesh(10)
        self.dim = 5
        self.V = dl.VectorFunctionSpace(mesh, "R", 0, dim=self.dim)
        
    def testSetLocal(self):
        z1 = dl.Function(self.V).vector()
        zt1 = AugmentedVector(z1)
        zt1_shared = AugmentedVector(z1, copy_vector=False)

        print("TESTING SET LOCAL")
        print("Initial vectors")
        print("zt1: ", zt1.get_local())
        print("zt1 (shared): ", zt1_shared.get_local())

        zt1_np = np.random.randn(self.dim + 1)
        zt1.set_local(zt1_np)
        print("After set local")
        print("zt1: ", zt1.get_local())
        print("zt1 (shared): ", zt1_shared.get_local())
        print("zt1 should be")
        print(zt1_np)
        print("zt1 (shared) should be zeros")
        self.assertTrue(np.allclose(zt1.get_local(), zt1_np))
        self.assertTrue(np.allclose(zt1_shared.get_local(), 0))

        zt1_shared_np = np.random.randn(self.dim + 1)
        zt1_shared.set_local(zt1_shared_np)
        print("After setting shared")
        print("zt1: ", zt1.get_local())
        print("zt1 (shared): ", zt1_shared.get_local())
        print("dolfin z: ", z1.get_local())

        self.assertTrue(np.allclose(zt1.get_local(), zt1_np))
        self.assertTrue(np.allclose(zt1_shared.get_local(), zt1_shared_np))
        self.assertTrue(np.allclose(z1.get_local(), zt1_shared_np[:-1]))


    def testAddLocal(self):
        z1 = dl.Function(self.V).vector()
        z2 = dl.Function(self.V).vector()
        zt1_np = np.random.randn(self.dim + 1)
        zt2_np = np.random.randn(self.dim + 1)
        zt1 = AugmentedVector(z1)
        zt2 = AugmentedVector(z2)
        zt1.set_local(zt1_np)
        zt2.set_local(zt2_np)

        print("TESTING ADD LOCAL")
        zt1.add_local(zt2.get_local())
        print("zt1: ", zt1.get_local())
        print("Sum: ", zt1_np + zt2_np)
        self.assertTrue(np.allclose(zt1.get_local(), zt1_np + zt2_np))
        

    def testAxpy(self):
        z1 = dl.Function(self.V).vector()
        z2 = dl.Function(self.V).vector()
        zt1_np = np.random.randn(self.dim + 1)
        zt2_np = np.random.randn(self.dim + 1)
        zt1 = AugmentedVector(z1)
        zt2 = AugmentedVector(z2)
        zt1.set_local(zt1_np)
        zt2.set_local(zt2_np)

        print("TESTING AXPY")
        a = 0.5 
        zt1.axpy(a, zt2)
        print("zt1: ", zt1.get_local())
        print("sum: ", zt1_np + a * zt2_np)
        self.assertTrue(np.allclose(zt1.get_local(), zt1_np + a * zt2_np))

    def testZero(self):        
        z1 = dl.Function(self.V).vector()
        zt1 = AugmentedVector(z1)
        zt1_np = np.random.randn(self.dim + 1)
        zt1.set_local(zt1_np)
        print("TESTING ZERO")
        print("Before zero, zt1: ", zt1.get_local())
        self.assertFalse(np.allclose(zt1.get_local(), 0)) 
        zt1.zero()
        print("After zero, zt1: ", zt1.get_local())
        self.assertTrue(np.allclose(zt1.get_local(), 0)) 


    def testGetVector(self):
        z1 = dl.Function(self.V).vector()
        z1.set_local(np.random.randn(self.dim))
        zt1 = AugmentedVector(z1)

        z1_get = zt1.get_vector()
        self.assertTrue(isinstance(z1_get, dl.PETScVector) or isinstance(z1_get, dl.Vector))
        self.assertTrue(np.allclose(z1.get_local(), z1_get.get_local())) 

    def testGetScalar(self):
        z1 = dl.Function(self.V).vector()
        zt1 = AugmentedVector(z1)

        zt1_np = np.random.randn(self.dim+1)
        zt1.set_local(zt1_np)
        t = zt1.get_scalar()
        self.assertEqual(t, zt1_np[-1])

    def testInner(self):
        z1 = dl.Function(self.V).vector()
        z2 = dl.Function(self.V).vector()
        zt1_np = np.random.randn(self.dim + 1)
        zt2_np = np.random.randn(self.dim + 1)
        zt1 = AugmentedVector(z1)
        zt2 = AugmentedVector(z2)
        zt1.set_local(zt1_np)
        zt2.set_local(zt2_np)

        ip = zt1.inner(zt2)
        self.assertTrue(np.allclose([ip], [np.inner(zt1_np, zt2_np)]))



if __name__ == "__main__":
    unittest.main()

