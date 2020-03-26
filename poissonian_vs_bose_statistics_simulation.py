#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:20:44 2019

@author: lukas
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def poss_distro_func(λ, k = 10):
    ks = np.arange(0,k+1,1)
    return ks, λ**ks/factorial(ks)*np.exp(-λ)

#Simulation

N = 1000
bin = 0.01

# simulate write photons
w_random_picks = np.random.rand(N)
w_hist, _ = np.histogram(w_random_picks, bins = np.arange(0, 1+bin, bin))

# simulate read photons
r_random_picks = np.random.rand(N)
r_hist, _ = np.histogram(r_random_picks, bins = np.arange(0, 1+bin, bin))

λ = N*bin
poss_ks, poss_distro = poss_distro_func(λ, k = λ*3)

plt.figure("Possonian distribution")
plt.clf()
plt.bar(poss_ks, poss_distro, label = 'Poissonian Distribution')
plt.hist(w_hist, histtype = 'step', density = True, color = 'k', align = 'left', bins = np.arange(0, 1+N, 1), label = 'Poissonian Write')
plt.hist(r_hist, histtype = 'step', density = True, color = 'r', align = 'left', bins = np.arange(0, 1+N, 1), label = 'Poissonian Read')
plt.xlim([-1,max(poss_ks)+4])
plt.legend()

#%%
pr = r_hist.mean()
pw = w_hist.mean()
prw = (w_hist*r_hist).mean()

g2 = prw/pr/pw # Cool!!!

#%% Playing around with bose-einstein distro

def  bose_distro_func(λ, k = 10):
    ks = np.arange(0,k+1,1)
    return ks, 1/(λ+1)*(λ/(λ+1))**ks

bose_ks, bose_distro = bose_distro_func(λ, k = λ*3)


#%% Simulate single-mode bose-einstein distro

intensity = np.linspace(0,10000,10000+1)
N_modes = 200

def phase(N_modes):
    phase_array = np.zeros(N_modes, dtype = np.complex)
    for i in range(N_modes):
        phase_array[i] = np.exp(2*np.pi*np.random.rand(1)*1j)
    return phase_array

for i in range(len(intensity)):
    phases = phase(N_modes)
    summed_phase = phases.sum()
    amplitude = np.sqrt(summed_phase.real**2 + summed_phase.imag**2)
    intensity[i] = amplitude**2
    
    
plt.figure('bose')
plt.plot(intensity)
plt.ylim((0,1.2*max(intensity)))

#%%

discrete_intensity = intensity.astype(int)
bose_hist, bins = np.histogram(discrete_intensity, bins = np.linspace(0,10000,10000+1))

bose_ks, bose_distro = bose_distro_func(200, k = 800)

plt.figure("Bose-Eisntein distribution")
plt.bar(bose_ks, bose_distro, label = 'Bose-Einstein Distribution')
plt.hist(discrete_intensity, histtype = 'step', density = True, color = 'r', align = 'left', bins = np.linspace(0,10000,10000+1), label = 'Bose-Eisntein Simulation')

#%%

pr = discrete_intensity.mean()
pw = discrete_intensity.mean()
prw = (discrete_intensity*np.roll(discrete_intensity,1)).mean()

g2 = prw/pr/pw # Cool!!!

print(g2)
