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


import time
import os, sys
import pickle
import argparse
import math

import numpy as np 
import matplotlib.pyplot as plt
import dolfin as dl
from matplotlib import cm
from matplotlib.colors import Normalize
from mpi4py import MPI

import hippylib as hp
import soupy


def get_target(target_case, target_param, comm_mesh=MPI.COMM_WORLD):
    """
    Get the dolfin expression for the target state
    """
    if target_case == "constant":
        u_expr = dl.Constant(target_param)
    
    elif target_case == "sinusoid":
        u_expr = dl.Expression("a*sin(kx*x[0])*sin(ky*x[1])", 
            a=target_param, kx=2*np.pi, ky=2*np.pi, degree=5, mpi_comm=comm_mesh)
    
    elif target_case == "arch":
        u_expr = dl.Expression("4*a*x[1]*(1-x[1])", a=target_param, degree=5, mpi_comm=comm_mesh)

    elif target_case == "smiley":
        # u_expr = SmileyExpression(a=target_param, degree=5, mpi_comm=comm_mesh)
        u_expr = SmileyExpression(a=target_param, degree=5)

    elif target_case == "frowney":
        # u_expr = FrowneyExpression(a=target_param, degree=5, mpi_comm=comm_mesh)
        u_expr = FrowneyExpression(a=target_param, degree=5)

    else:
        raise ValueError("incorrect target case")

    return u_expr


class FrowneyExpression(dl.UserExpression):
    def __init__(self, a, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.eye_width = 0.3
        self.mouth_length = 1 
        self.mouth_width = 0.05
        self.mouth_radius = 0.3

        self.mouth_degrees = 90
        self.DEGREES2RADIANS = np.pi/180
        self.mouth_radians = self.mouth_degrees * self.DEGREES2RADIANS
        self.mouth_max = self.mouth_degrees * self.DEGREES2RADIANS + self.mouth_radians
        self.mouth_min = self.mouth_degrees * self.DEGREES2RADIANS - self.mouth_radians
        self.mouth_start = 90 * self.DEGREES2RADIANS

        self.left_center = np.array([0.3, 0.7])
        self.right_center = np.array([0.7, 0.7])
        self.center = np.array([0.5, 0.1])

        self.BUMP_SCALE = 8

    def _polar(self, x):
        d = x - self.center
        theta = np.arctan2(d[1], d[0])
        rho = np.linalg.norm(d, 2)
        return rho, theta
        
    def _bump(self, t):
        BUMP_SCALE = 8
        if np.abs(t) < 1:
            return np.exp(BUMP_SCALE) * np.exp(- BUMP_SCALE/(1 - t**2))
        else:
            return 0.0

    def _mouth_bump(self, t):
        BUMP_SCALE = 1
        if np.abs(t) < 1:
            return np.exp(BUMP_SCALE) * np.exp(- BUMP_SCALE/(1 - t**2))
        else:
            return 0.0

    def _gaussian(self, t):
        return np.exp(-t**2/2)

    def _rescale_distance(self, x):
        left_dist = np.linalg.norm(x - self.left_center)/self.eye_width
        right_dist = np.linalg.norm(x - self.right_center)/self.eye_width
        return left_dist, right_dist

    def _rescale_angle(self, theta):
        return (theta - self.mouth_start)/self.mouth_radians

    def eval(self, value, x):
        # left_eye = np.exp(-np.inner(x - self.left_center, x - self.left_center)/(2*self.eye_width**2))
        # right_eye = np.exp(-np.inner(x - self.right_center, x - self.right_center)/(2*self.eye_width**2))
        left_d, right_d = self._rescale_distance(x)
        left_eye = self._bump(left_d)
        # left_eye = self._gaussian(left_d)
        right_eye = self._bump(right_d)
        # right_eye = self._gaussian(right_d)

        rho, theta = self._polar(x)
        mouth = np.exp(-(rho - self.mouth_radius)**2/(2*self.mouth_width**2)) * self._mouth_bump(self._rescale_angle(theta))
        # bc = x[1] * 0
        # value[0] = self.a*left_eye + self.a*right_eye + self.a*mouth + bc
        value[0] = self.a*left_eye + self.a*right_eye + self.a*mouth

    def value_shape(self):
        return ()



class SmileyExpression(dl.UserExpression):
    def __init__(self, a, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.eye_width = 0.3
        self.mouth_length = 1 
        self.mouth_width = 0.05
        self.mouth_radius = 0.3

        self.mouth_degrees = 90
        self.DEGREES2RADIANS = np.pi/180
        self.mouth_radians = self.mouth_degrees * self.DEGREES2RADIANS
        self.mouth_max = self.mouth_degrees * self.DEGREES2RADIANS + self.mouth_radians
        self.mouth_min = self.mouth_degrees * self.DEGREES2RADIANS - self.mouth_radians
        self.mouth_start = -90 * self.DEGREES2RADIANS

        self.left_center = np.array([0.3, 0.7])
        self.right_center = np.array([0.7, 0.7])
        self.center = np.array([0.5, 0.5])

        self.BUMP_SCALE = 8

    def _polar(self, x):
        d = x - self.center
        theta = np.arctan2(d[1], d[0])
        rho = np.linalg.norm(d, 2)
        return rho, theta
        
    def _bump(self, t):
        BUMP_SCALE = 8
        if np.abs(t) < 1:
            return np.exp(BUMP_SCALE) * np.exp(- BUMP_SCALE/(1 - t**2))
        else:
            return 0.0

    def _mouth_bump(self, t):
        BUMP_SCALE = 1
        if np.abs(t) < 1:
            return np.exp(BUMP_SCALE) * np.exp(- BUMP_SCALE/(1 - t**2))
        else:
            return 0.0

    def _gaussian(self, t):
        return np.exp(-t**2/2)

    def _rescale_distance(self, x):
        left_dist = np.linalg.norm(x - self.left_center)/self.eye_width
        right_dist = np.linalg.norm(x - self.right_center)/self.eye_width
        return left_dist, right_dist

    def _rescale_angle(self, theta):
        return (theta - self.mouth_start)/self.mouth_radians

    def eval(self, value, x):
        # left_eye = np.exp(-np.inner(x - self.left_center, x - self.left_center)/(2*self.eye_width**2))
        # right_eye = np.exp(-np.inner(x - self.right_center, x - self.right_center)/(2*self.eye_width**2))
        left_d, right_d = self._rescale_distance(x)
        left_eye = self._bump(left_d)
        # left_eye = self._gaussian(left_d)
        right_eye = self._bump(right_d)
        # right_eye = self._gaussian(right_d)

        rho, theta = self._polar(x)
        mouth = np.exp(-(rho - self.mouth_radius)**2/(2*self.mouth_width**2)) * self._mouth_bump(self._rescale_angle(theta))
        # bc = x[1] * 0
        # value[0] = self.a*left_eye + self.a*right_eye + self.a*mouth + bc
        value[0] = self.a*left_eye + self.a*right_eye + self.a*mouth

    def value_shape(self):
        return ()



def plot_sources(z_np, N_wells_per_side, lower=0.25, upper=0.75):
    well_grid = np.linspace(lower, upper, N_wells_per_side)
    X, Y = np.meshgrid(well_grid, well_grid)
    bottom = np.zeros(X.shape)

    Z = np.zeros(Y.shape)

    count = 0
    for i in range(N_wells_per_side):
        for j in range(N_wells_per_side):
            Z[j,i] = z_np[count]
            count += 1

    width = (upper-lower)/N_wells_per_side/2
    depth = width

    z_max = np.max(np.abs(Z))

    cmap = cm.get_cmap('coolwarm')
    norm = Normalize(vmin=-z_max, vmax=z_max)
    colors = cmap(norm(Z.ravel()))


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, -125)
    ax.bar3d(X.ravel(), Y.ravel(), bottom.ravel(), width, depth, Z.ravel(), color=colors)
    ax.set_xlabel("$x$", labelpad=12)
    ax.set_ylabel("$y$", labelpad=12)
    ax.set_zlabel("$z$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    z_max = np.max(np.abs(z_np))
    ax.set_zlim(-1.1*z_max, 1.1*z_max)
    return fig, ax
