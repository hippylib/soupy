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

"""
macOS-specific configuration for FEniCS/DOLFIN JIT compilation.

On newer macOS SDKs (11+), the FFC JIT compiler requires C++17 and specific
flags to work correctly with Apple's Clang and libc++.

Usage:
    # Import this BEFORE importing dolfin
    from soupy.utils.macos_config import configure_macos_compiler
    configure_macos_compiler()

    import dolfin as dl
    # ... rest of your code
"""

import os
import sys


def configure_macos_compiler():
    """
    Configure the C++ compiler for FEniCS JIT compilation on macOS.

    This function:
    1. Sets DOLFIN_JIT_CXXFLAGS to include -std=c++17
    2. Sets CC and CXX to use the system clang

    This is only applied on macOS (darwin) systems.
    Must be called BEFORE importing dolfin.
    """
    if sys.platform != "darwin":
        return

    # Set C++17 flag for DOLFIN JIT compilation
    jit_flag = "-std=c++17"
    existing = os.environ.get("DOLFIN_JIT_CXXFLAGS", "")
    if jit_flag not in existing:
        os.environ["DOLFIN_JIT_CXXFLAGS"] = (existing + " " + jit_flag).strip()

    # Use system clang
    os.environ["CXX"] = "/usr/bin/clang++"
    os.environ["CC"] = "/usr/bin/clang"


def configure_dolfin_form_compiler(dl):
    """
    Configure dolfin form compiler parameters for macOS.

    This function sets the cpp_optimize_flags to include C++17 and
    disable availability macros.

    Must be called AFTER importing dolfin.

    Args:
        dl: The dolfin module (pass after importing)
    """
    if sys.platform != "darwin":
        return

    dl.parameters["form_compiler"]["cpp_optimize"] = True
    fc_flags = dl.parameters["form_compiler"]["cpp_optimize_flags"]
    extra_flags = "-std=c++17 -D_LIBCPP_DISABLE_AVAILABILITY"
    if extra_flags not in fc_flags:
        dl.parameters["form_compiler"]["cpp_optimize_flags"] = fc_flags + " " + extra_flags
