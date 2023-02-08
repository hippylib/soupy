import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp 
import dolfin as dl 
import numpy as np

class UniformDistribution:
    """
    Class for sampling from a uniform distribution to `dl.Vector`
    """
    def __init__(self, Vh, a, b):
        """ 
        Constructor:
            :code: `Vh`: Function space for sample vectors
            :code: `a`: Lower bound
            :code: `b`: Upper bound
            :code: `ndim`: Dimension of sample vectors
        """
        self.Vh = Vh
        self.a = a
        self.b = b
        self.ndim = self.Vh.dim()
        self._dummy = dl.Function(Vh).vector()

    def init_vector(self, v):
        v.init( self._dummy.local_range() )

    def sample(self, out):
        assert out.mpi_comm().Get_size() == 1
        v = np.random.rand(self.ndim) * (self.b-self.a) + self.a
        out.set_local(v)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    DIM = 4
    NS = 1000
    mesh = dl.UnitSquareMesh(20, 20)
    Vh = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=DIM)

    a = -1
    b = 2
    control_dist = UniformDistribution(Vh, a, b)

    z = dl.Vector()
    print(z)
    control_dist.init_vector(z)
    
    samples = []
    for i in range(NS):
        control_dist.sample(z)
        samples.append(z.get_local())
    samples = np.array(samples)
    
    plt.figure()
    for i in range(DIM):
        plt.subplot(221+i)
        plt.hist(samples[:,i])
    plt.show()
