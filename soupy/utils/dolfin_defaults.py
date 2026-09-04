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

import os


def configure_dolfin_defaults():
    """
    Apply SOUPy-wide dolfin defaults if dolfin is available.

    Environment overrides:
      - SOUPY_QUADRATURE_DEGREE: integer to override the default quadrature degree
      - SOUPY_DISABLE_DEFAULT_QUADRATURE: set to 1/true/yes to disable setting it
    """
    try:
        import dolfin as dl
    except Exception:
        return

    disable = os.environ.get("SOUPY_DISABLE_DEFAULT_QUADRATURE", "").strip().lower()
    if disable in {"1", "true", "yes"}:
        return

    qdeg_env = os.environ.get("SOUPY_QUADRATURE_DEGREE", "").strip()
    if qdeg_env:
        try:
            qdeg = int(qdeg_env)
        except ValueError:
            return
    else:
        qdeg = 4

    try:
        dl.parameters["form_compiler"]["quadrature_degree"] = qdeg
    except Exception:
        pass
