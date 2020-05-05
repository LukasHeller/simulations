# Add refractive index in sympy code...

import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.optics import BeamParameter, \
    ThinLens, FreeSpace, \
    CurvedMirror, FlatMirror, \
    CurvedRefraction, FlatRefraction
    
# from sympy.physics.optics import BeamParameter
        
import itertools

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def omega(p, z, focus,n):
    """
    Waist of the beam

    Parameters
    ----------
    p : p(z)
        complex parameter incorporating waist, curvature and wavelength
    z : length (m)
        position along the axis
    focus : length (m)
        Position of the focal spot

    Returns
    -------
    array of beam waist

    """
    w_0 = float(p.w_0)/np.sqrt(n)
    print(w_0)
    return w_0*np.sqrt(1+((z-focus)/float(p.z_r))**2)

def plotting(ax,helper_dist,p,n,lc,label):
    """
    Plots the wasit of the beam as it propagates along the z-axis

    Parameters
    ----------
    ax : axis
        axis to plot onto
    helper_dist : array (m)
        array of positions of objects, including Zero and End
    p : p(z)
        complex parameter incorporating waist, curvature and wavelength
    lc : color
        line color
    label : string, float
        label of this line

    Returns
    -------
    None.

    """
    for i, prop in enumerate(p):
        z = np.linspace(helper_dist[i], helper_dist[i+1], 200)
        focus = helper_dist[i] - float(prop.z)
        ax.plot(z*1e2, omega(prop,z,focus,n[i])*1e6, lc)
        ax.plot(z*1e2, -omega(prop,z,focus,n[i])*1e6, lc)
        
    ax.lines[-1].set_label(label)
    ylim = ax.get_ylim()
    
    for i, prop in enumerate(p):
        ax.axvline(helper_dist[i]*1e2, color = 'grey')
        ax.fill_between((helper_dist[i]*1e2,helper_dist[i+1]*1e2),
                        (1e4,1e4),(-1e4,-1e4), alpha = 1-1/n[i], color = "b")
    
    ax.set_ylim((ylim[0]*1.2, ylim[1]*1.2))
    
def propagate(param, ax,lc,label):
    """
    Propagation of a Gaussian beam along z.

    Parameters
    ----------
    param : dict
        parameter set containing position, transfer matrixes and initial beam parameter
    ax : axis
        axis to plot onto
    lc : tuple
        line color
    label : string, float
        label of this line

    Returns
    -------
    p : dict
        initial parameter set "param", with list of 
        propagated complex beam parameters appended

    """
    
    p = param.copy()
    p["p"] = [p["p_init"]]
    helper_dist = [p["plot_range"][0]] + p["dist"]
    rel_dist = [helper_dist[i+1]-helper_dist[i] for i in range(len(helper_dist)-1)]
    
    
    for d,obj in zip(rel_dist,p["objects"]):
        new = obj*FreeSpace(d)*p["p"][-1]
        p["p"].append(BeamParameter(new.wavelen, float(new.z), w=float(new.w_0)))
    
    
    helper_dist = helper_dist + [p["plot_range"][1]]     
    plotting(ax, helper_dist, p["p"], p["refr"], lc, label)
    
    # This cannot be done. Must change the x_ticks instead, otherwise
    # zoom functionality does not work properly
    
    # ax.set_xticklabels(["{:.0f}".format(x) for x in ax.get_xticks()*1e2])
    # ax.set_yticklabels(["{:.0f}".format(y) for y in ax.get_yticks()*1e+6])
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel(r"Waist ($\mu$m")
    ax.set_xlim((p["plot_range"][0]*1e2,p["plot_range"][1]*1e2))
    ax.legend()
    
    return p

#%% Example calculation

# Define a scneary with a couple of transfer matrixes and
# corresponding positions
dist = [0.3,0.6, 1, 1.2,1.3]
f1,f2,f3 = 0.2, 0.1, 0.6
n1,n2 = 1,2
objects = [ThinLens(f1),ThinLens(f2),ThinLens(f3),CurvedRefraction(0.3,n1,n2),CurvedRefraction(-0.3,n2,n1)]
n = [n1,n1,n1,n1,n1,n1]

n1,n2 = 1,1          # Some refractive indices
waist = 0.05e-3         # Beam waist (m)
dist = [0.2]   # Position of objects, array of length len(dist)
plot_range = (0,0.5)     # Lotting range, should be biger than "dist"
n = [n1,n1,n2,n1]      # Refractive index, array of length len(dist)+1
# Objects at the positions given by "dist"
objects = [ThinLens(0.1)]

param = {"dist": dist,
         "objects": objects,
         "p_init": BeamParameter(780e-9,0, w = waist),
         "refr": n,
         "plot_range": plot_range}

# Figure and axis to plot ontp
fig, ax = plt.subplots(num = 1)
# an iterator for the colors
c = itertools.cycle(colors)

# Loop through some parameter set
# for f1 in [.05, .1, .15, .2,2]:
#for f1 in [1]:
param_c = param.copy()
# param_c["objects"][0] = ThinLens(f1)
p_propagated = propagate(param, ax, next(c), f1)