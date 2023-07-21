import numpy as np 
from mpi4py import MPI 

def allocate_process_sample_sizes(sample_size, comm_sampler):
    """
    Compute the number of samples needed in each process 
    Return result as a list 

    :param sample_size: Total number of samples
    :type sample_size: int 
    :param comm_sampler: MPI communicator for sample allocation 
    :type comm_sampler: :py:class:`mpi4py.MPI.Comm`
    """ 
    n, r = divmod(sample_size, comm_sampler.size)
    sample_size_allprocs = []
    for i_rank in range(comm_sampler.size):
        if i_rank < r: 
            sample_size_allprocs.append(n+1)
        else:
            sample_size_allprocs.append(n)
    return sample_size_allprocs
