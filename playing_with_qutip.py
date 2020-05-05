#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:47:06 2020

@author: lukas
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,10,100000)
w_0 = 10000
w_m = 10
M = 10
E = np.exp(1j*(w_0*t + M*np.sin(w_m*t)))
# E = np.exp(1j*(w_0*t))
# plt.plot(t, np.abs(E)**2)

plt.figure(1)
plt.plot(t,E.real)

plt.figure(2)
plt.plot(np.fft.fft(E.real))



#%%

import fdtd

grid = fdtd.Grid(
    shape = (25e-6, 15e-6, 1), # 25um x 15um x 1 (grid_spacing) --> 2D FDTD
)

print(grid)

grid[11:32, 30:84, 0] = fdtd.Object(permittivity=1.7**2, name="object")
grid[13e-6:18e-6, 5e-6:8e-6, 0] = fdtd.Object(permittivity=1.5**2)

grid[7.5e-6:8.0e-6, 11.8e-6:13.0e-6, 0] = fdtd.LineSource(
    period = 1550e-9 / (3e8), name="source"
)

grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

grid.run(total_time=10)

grid.visualize(z=0)