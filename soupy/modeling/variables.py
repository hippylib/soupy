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

"""
Enumerator for the variables of the inverse problem:
 - :code:`STATE = 0` for the state variable
 - :code:`PARAMETER = 1` for the parameter variable
 - :code:`ADJOINT = 2` for the adjoint variable
 - :code:`CONTROL = 3` for the control variable
"""
STATE= 0
PARAMETER = 1
ADJOINT = 2
CONTROL = 3