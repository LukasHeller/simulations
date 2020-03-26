# -*- coding: utf-8 -*-
"""
This file simulates cross- and autocorrelation functions of the two mode squeezed state in the DLCZ memory

@author: lheller
"""

import numpy as np
import matplotlib.pyplot as plt

#%% 

p = np.linspace(0.001,0.999, num = 999)
n_mean = p/(1-p)   # Mean Photon Number of the Two Mode Squeezed State
n_coincidence = p*(1+p)/(1-p)**2
g2_S_AS = 1+1/p

plt.semilogy(p,n_mean, label = 'Mean Photon Number')
plt.semilogy(p,n_coincidence, label = 'Mean Coincidences')
plt.semilogy(p,g2_S_AS, label = 'g2')
plt.legend()
plt.grid()

