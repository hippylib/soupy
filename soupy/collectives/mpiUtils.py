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


def allgather_vector_as_numpy(v):
    """
    All gather a dolfin vector :code:`v` 
    such that each process has a full copy of the numpy 
    array representing v 

    :param v: Vector to gather
    :type v: dl.Vector 

    :returns: A numpy vector :code:`v_np` with the full data of :code:`v` on every process
    """
    mpi_comm = v.mpi_comm()
    vector_size = v.size()
    v_np = v.gather_on_zero()
    if mpi_comm.Get_rank() > 0:
        v_np = np.zeros(vector_size, dtype=np.float64)
    mpi_comm.Bcast(v_np, root=0)
    return v_np 

def set_local_from_global(v, v_np):
    """
    Set the local components of a dolfin vector :code:`v` 
    from a global numpy array :code:`v_np` using its local range 

    :param v: Vector to set
    :type v: dl.Vector 
    :param v_np: numpy array for global entries
    :type v: np.ndarray
    """
    
    local_range = v.local_range()
    if len(local_range) > 0:
        v.set_local(v_np[local_range[0] : local_range[1]])
    v.apply("")

def get_global(v):
    """
    Retrieves the global representation of :code:`v` as a numpy array
    """
    mpi_size = v.mpi_comm().Get_size()
    if mpi_size == 1:
        return v.get_local()
    else:
        return allgather_vector_as_numpy(v)












