"""
Cavity calculations
Based on the book Lasers by Siegmann
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as con

import seaborn as sns
sns.set() 
sns.set_context("paper")

c = con.c                       # Speed of light (m/s)

# #%% Definition of reflectivities and losses
# # DLCZ cavity

# p = {'I_inc': 1,                # Input power (W)
#      'R_in': 0.87,              # Reflectivity of input mirror (%)
#      'R_out': 0.999,            # Reflectivity of output mirror (%)
#      'R_n':[0.999],             # Additional mirrors (%)
     
#      'p': np.array([0.8]),     # Effective cavity length (equals 2L for linear cavity) (m)
#      'n': np.array([1]),      # Refractive index of medium
#      'a': np.array([0]),        # Absorption coefficient of medium (1/m)
# #     'R': np.array([0]),        # All mirror refl, including R_in and R_out (%)
#      'L': np.array([0.11]),        # Extra losses per roundtrip (%)
     
#      'p_0': np.array([0.4]),   # Effective first cavity leg (L for linear cavity) (m)
#      'n_0': np.array([1]),    # Refractive index on first leg
#      'a_0': np.array([0]),      # Absorption coefficient of medium on first leg (1/m)
#      'R_0': np.array([0.999]),       # Mirrors on first leg (%)
#      'L_0': np.array([0.05]),      # Extra losses on first leg (%)
     
#      'lambda': 780e-9,          # Central wavelength (m)
#      'high_res': False}         # High resolution (only for low finesse)

#%% Definition of reflectivities and losses
# new FP filtering cavity

p = {'I_inc': 1,                # Input power (W)
     'R_in': 0.99,              # Reflectivity of input mirror (%)
     'R_out': 0.99,             # Reflectivity of output mirror (%)
     'R_n':[],                  # Additional mirrors (%)
     
     # 'p': np.array([16e-3]),    # Effective cavity length (equals 2L for linear cavity) (m)
     # 'n': np.array([1.53]),     # Refractive index of medium
     # 'a': np.array([1000e-4]),    # Absorption coefficient of medium (1/m)
     # 'L': np.array([2*20e-6 + 2*13e-6 + 2*1e-6]),    # Extra losses per roundtrip (%)
     
     'p': np.array([16e-3]),    # Effective cavity length (equals 2L for linear cavity) (m)
     'n': np.array([1.45]),     # Refractive index of medium
     'a': np.array([1000e-4]),  # Absorption coefficient of medium (1/m)
     'L': np.array([200e-6]),   # Extra losses per roundtrip (%)
     
     'p_0': np.array([]),   # Effective first cavity leg (L for linear cavity) (m)
     'n_0': np.array([]),    # Refractive index on first leg
     'a_0': np.array([]),      # Absorption coefficient of medium on first leg (1/m)
     'R_0': np.array([]),       # Mirrors on first leg (%)
     'L_0': np.array([]),      # Extra losses on first leg (%)
     
     'lambda': 780.001e-9,          # Central wavelength (m)
     'high_res': False}         # High resolution (only for low finesse)

#%% Updates cavity parameters. Depend on the definition of parameters p above
def p_update(p):
    p_temp = p.copy()
    
    p_temp['R'] = np.concatenate((np.array([p['R_in'],p['R_out']]),p_temp['R_n']))
    p_temp['d_R'] = sum(np.log(1/p_temp['R']))                  # Delta notation of R
    # Delta notation of losses
    p_temp['d_L'] = sum(np.log(1/(1-p_temp['L']))) + 2*sum([p*a for p, a in zip(p_temp['p'], p_temp['a'])])
    p_temp['d_c'] = p_temp['d_R'] + p_temp['d_L']               # Effective losses combined
    p_temp['g_rt_2'] = np.exp(-p_temp['d_c'])                   # Real loss term squared
    # Effective cavity length
    p_temp['p_eff'] = sum([p*n for p,n in zip(p_temp['p'], p_temp['n'])]) 
    
    p_temp['d_R_0'] = sum(np.log(1/p_temp['R_0']))                  # Delta notation of R
    # Delta notation of losses
    p_temp['d_L_0'] = sum(np.log(1/(1-p_temp['L_0']))) + 2*sum([p_0*a_0 for p_0, a_0 in zip(p_temp['p_0'], p_temp['a_0'])])
    p_temp['d_c_0'] = p_temp['d_R_0'] + p_temp['d_L_0']               # Effective losses combined
    p_temp['g_rt_2_0'] = np.exp(-p_temp['d_c_0'])                   # Real loss term squared
    # Effective cavity length
    p_temp['p_eff_0'] = sum([p_0*n_0 for p_0,n_0 in zip(p_temp['p_0'], p_temp['n_0'])]) 
    
    p_temp['E_inc'] = np.sqrt(p_temp['I_inc'])                  # Incident electric field
        
    return p_temp

# Generates the different parameters needed for the 
# transmission, reflection and field enhancement of the cavity
p = p_update(p)

#%%
f_min = c/780.12e-9 # Minimum frequency
f_max = c/780e-9     # Maximum frequency

# List of frequencies*2pi between f_min and f_max
omega = np.linspace(2*np.pi*f_min,2*np.pi*f_max,1000)
omega = np.linspace(0,2*np.pi*30e9,10000)

def fields(p, omega):
    g_omega = np.exp(-1j*omega*p['p_eff']/c)
    E_circ =   1j*(1-p['R_in'])**(1/2)/(1-p['g_rt_2']**(1/2)*g_omega)*p['E_inc']
    E_circ_per_E_launch = 1/(1-p['g_rt_2']**(1/2)*g_omega)
    E_trans =  1j*(1-p['R_out'])**(1/2)*E_circ*p['g_rt_2_0']**(1/2)
    E_refl =   1/p['R_in']**(1/2)*(p['R_in'] -p['g_rt_2']**(1/2)*g_omega)/(1- p['g_rt_2']**(1/2)*g_omega)
    return [E_circ, E_trans, E_refl,E_circ_per_E_launch]

def cav_params(p):
    params = {}
    params['FSR'] = c/p['p_eff']
    params['q'] = p['p_eff']/p['lambda']
    params['F'] = np.pi*p['g_rt_2']**(1/4)/(1-p['g_rt_2']**(1/2))
    params['FWHM'] = 2*c/np.pi/p['p_eff']*np.arcsin((1-p['g_rt_2']**(1/2))/(2*p['g_rt_2']**(1/4)))
    # next one is wrong. does not contain the alpha
    params['Esc_eff_trans'] = (1-p['R_out'])/(sum(1-p['R'])+sum(p['L']))
    params['Esc_eff_back'] = (1-p['R_in'])/(sum(1-p['R'])+sum(p['L']))
    params['Intra_cav_enh'] = 2*params['F']/np.pi
    
    print("--- Cavity characteristics ---")
    for i in params:
        print("{:<8}: {:.2f}".format(i, params[i]))
    print("-------------------------") 
    
    return params

#%% Plotting
    
plt.close('all')
fig,ax = plt.subplots(1,4, figsize = (12,3), num = 1)
para = "p"

for scanner in [np.array([0.02]),
                np.array([0.012]),
                np.array([0.008])]:
    p_temp = p.copy()
    p_temp[para] = scanner
    
    if not p_temp['high_res']:
        p_temp["p_0"] = p_temp["p"]/2
        p_temp["n_0"] = p_temp["n"]
        p_temp["a_0"] = p_temp["a"]
        p_temp["R_0"] = np.array([])
        p_temp["L_0"] = p_temp["L"]/2
    
    p_temp = p_update(p_temp)
    label = str(scanner)
    
    E_circ, E_trans, E_refl, E_circ_per_E_launch = fields(p_temp, omega)
    cav_params(p_temp)
    
    ax[0].semilogy(omega/2/np.pi*1e-9, np.abs(E_circ/p['E_inc'] )**2, label =  label)
    ax[1].semilogy(omega/2/np.pi*1e-9, np.abs(E_trans/p['E_inc'] )**2, label = label)
    ax[2].semilogy(omega/2/np.pi*1e-9, np.abs(E_refl/p['E_inc'] )**2, label = label)
    ax[3].semilogy(omega/2/np.pi*1e-9, 1-np.abs(E_refl/p['E_inc'] )**2 - np.abs(E_trans/p['E_inc'] )**2, label = label)

ax[0].set_title( "Circulating")
ax[0].set_ylabel(r"$I_{circ}/I_{inc}$")
ax[0].legend(title = para)

ax[1].set_title( "Transmitted")
ax[1].set_ylabel(r"$I_{trans}/I_{inc}$")
ax[1].legend(title = para)

ax[2].set_title( "Reflected")
ax[2].set_ylabel(r"$I_{refl}/I_{inc}$")
ax[2].legend(title = para)

ax[3].set_title( "Intracavity Losses")
ax[3].set_ylabel(r"$I_{loss}/I_{inc}$")
ax[3].legend(title = para)

plt.tight_layout()

