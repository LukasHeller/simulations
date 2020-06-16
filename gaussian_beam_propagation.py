import numpy as np
import matplotlib.pyplot as plt
import itertools
import gaussopt
from gaussopt import RayTransferMatrix, BeamParameter, \
    ThinLens, FreeSpace, \
    CurvedRefraction, FlatRefraction

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
    
def omega(p, z, focus):
    """
    Waist of the beam along z.

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
    return p.w_0*np.sqrt(1+((z-focus)/p.z_r)**2)

def R(p, z, focus):
    """
    Radius of Curvature of the beam along z.

    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    focus : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    print(focus)
    print(p.z_r)
    return (z-focus)*(1+(p.z_r/(z-focus))**2)

def plotting(axes,helper_dist,p,lc,label):
    """
    Plots the waist of the beam as it propagates along the z-axis

    Parameters
    ----------
    axes : array of two axis, for waist and radius
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
        focus = helper_dist[i] - prop.z
        axes[0].plot(z*1e2, omega(prop,z,focus)*1e6, color = lc)
        axes[0].plot(z*1e2, -omega(prop,z,focus)*1e6, color = lc)
        import pdb
        # pdb.set_trace()
        axes[1].plot(z*1e2, R(prop,z,focus), color = lc, ls = ":")
        
        print(f"--- {helper_dist[i]*1e2} to {helper_dist[i+1]*1e2} ---")
        print(f"w_0: {prop.w_0*1e6} um")
        print(f"z_r: {prop.z_r*1e2} cm")
        print()
        
    for ax in axes:
        ax.lines[-1].set_label(label)
        ylim = ax.get_ylim()
        
        for i, prop in enumerate(p):
            ax.axvline(helper_dist[i]*1e2, color = 'grey')
            ax.fill_between((helper_dist[i]*1e2,helper_dist[i+1]*1e2),
                            (1e4,1e4),(-1e4,-1e4), alpha = 1-1/prop.n, color = "b")
        
        ax.set_ylim((ylim[0]*1.2, ylim[1]*1.2))
        

    
def propagate(param, ax,lc,label = "l"):
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
        p2 = obj*FreeSpace(d)*p["p"][-1]
        p2.n = obj.n2
        p["p"].append(p2)
    
    helper_dist = helper_dist + [p["plot_range"][1]]     
    plotting(ax, helper_dist, p["p"], lc, label)
    
    axes[0].set_ylabel(r"Waist ($\mu$m)")
    axes[0].set_xlim((p["plot_range"][0]*1e2,p["plot_range"][1]*1e2))
    axes[0].legend()
    
    axes[1].set_xlabel("Distance (cm)")
    axes[1].set_ylabel("Radius (m)")
    axes[1].set_xlim((p["plot_range"][0]*1e2,p["plot_range"][1]*1e2))
    axes[1].legend()
    
    return p

#%% Example calculation

# Figure and axis to plot ontp
fig, axes = plt.subplots(2,1,num = 1)
# an iterator for the colors
c = itertools.cycle(colors)

# Define a scneary with a couple of transfer matrixes and
# corresponding positions

f1,f2 = 0.35, 0.15
plot_range = (0,0.05)     # Plotting range, should be biger than "dist"


# dist = [0.008]
# n_glass = 1.43
# objects = [CurvedRefraction(-0.2, n_glass,1)]
# plot_range = (0,0.02)     # Lotting range, should be biger than "dist"
# beamPara = BeamParameter(780e-9, 0, n_glass, w = 100e-6)

dist = [0.1]
objects = [ThinLens(f1)]
beamPara = BeamParameter(780e-9, 0, 1, w = 800e-6)

param1 = {"dist": dist,
          "objects": objects,
          "p_init": beamPara,
          "plot_range": plot_range}

dist = [0.1]
objects = [ThinLens(f1)]
beamPara = BeamParameter(780e-9, 0, 1, w = 1200e-6)

param2 = {"dist": dist,
          "objects": objects,
          "p_init": beamPara,
          "plot_range": plot_range}

dist = []
objects = []
beamPara = BeamParameter(780e-9, 0, 1, w = 1200e-6)

param = {"dist": dist,
          "objects": objects,
          "p_init": beamPara,
          "plot_range": plot_range}

scanThis = [BeamParameter(780e-9, 0, 1.45, w = 57e-6),
            BeamParameter(780e-9, 0, 1.5, w = 45e-6),
            BeamParameter(780e-9, 0, 1.5, w = 58e-6),]

para = "p_init"
for scanner in scanThis:
    param_c = param.copy()
    param_c[para] = scanner
    print(scanner.w_0)
    p_propagated = propagate(param_c, axes, next(c), label =scanner.w_0)
    
#%%
# Quanteaser
thickness = 0.0053
R = 0.075
n = 1.45

# ours 1 thick
thickness = 0.008
R = 0.06
n = 1.5

# ours 1 thin
# thickness = 0.004
# R = 0.04
# n = 1.5

# new
thickness = 0.00635
R = 0.1
n = 1.45


z_r = thickness*np.sqrt(R/thickness-1)
w_0 = np.sqrt(z_r*780e-9/n/np.pi)
print(f"Raleighlength is {z_r*1e2:.02f} cm")
print(f"Beamsize at z=0 is {w_0*1e6:.02f} um")

#%%

z = np.linspace(0,10,100)
z_r_list  = np.linspace(0.11,1,10)
for z_r in z_r_list:
    plt.plot(z, z*(1+(z_r/z)**2))

