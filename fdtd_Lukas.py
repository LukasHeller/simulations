import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con
import matplotlib.cm as cm

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def fdtd_1d(eps_rel, dx, time_span,source_frequency, source_position,source_pulse_length):
    """
    Computes the temporal evolution of a pulsed excitation using the 1D FDTD method.
    The temporal center of the pulse is placed at a simulation time of
    3*source_pulse_length. The origin x=0 is in the center of the computational domain.
    All quantities have to be specified in SI units.
    Arguments:
        
    eps_rel: rel. permittivity distribution within the computational domain (vector)
    dx: spacing of the simulation grid (scalar, please ensure dx <= lambda/20)
    time_span: time span of simulation (scalar)
    source_frequency: frequency of current source (scalar)
    source_position: spatial position of current source (scalar)
    source_pulse_length : temporal width of Gaussian envelope of the source (scalar)
    
    Returns:
    
    Ez: z component of E(x,t) (matrix, t varies along columns)
    Hy: y component of H(x,t) (matrix, t varies along columns)
    x: spatial coordinates of the field output (vector)
    t: time of the field output (vector)
    """
    
    c = con.speed_of_light
    mu_0 = con.mu_0
    eps_0 = con.epsilon_0
    
    dt = dx/2/c
    t = np.arange(0,time_span,dt)
    
    N_x = len(eps_rel)
    N_t = len(t)
    
    x = np.arange(0,N_x)*dx
    
    Ez = np.zeros((N_x+1,N_t))
    Hy = np.zeros((N_x,N_t))
    
    V_x = c/np.sqrt(eps_rel)
    
    print("Ez: {:.2f} Mbytes".format(float((Ez.size * Ez.itemsize))*1e-6))
    
    # import pdb
    # pdb.set_trace()
    
    printProgressBar(0, N_t, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i in range(N_t-1):
        Ez[1:-1,i+1] = Ez[1:-1,i] + 1/(eps_0*eps_rel[1:])*dt/dx*(Hy[1:,i]-Hy[:-1,i])

        j = 1/(eps_0*eps_rel[100])*dt/dx* \
            np.exp(-((t[i] - 3*source_pulse_length )/source_pulse_length)**2)* \
            np.cos(source_frequency*t[i])
        
        Ez[0,i+1] = Ez[1,i] + (V_x[0]*dt-dx)/(V_x[0]*dt+dx)*(Ez[1,i+1]-Ez[0,i])
        Ez[-1,i+1] = Ez[-2,i] + (V_x[-1]*dt-dx)/(V_x[-1]*dt+dx)*(Ez[-2,i+1]-Ez[-1,i])

        Ez[100,i+1] = Ez[100,i+1] + j
        # Hy[:,i+1] = Hy[:,i] + 1/mu_0*dt/dx*(Ez[1:,i+1]-Ez[:-1,i+1])
        Hy[:,i+1] = Hy[:,i] + 1/mu_0*dt/dx*(Ez[1:,i+1]-Ez[:-1,i+1])
        printProgressBar(i, N_t, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    Ez = Ez[:-1]
    return Ez, Hy,x,t
    
def plotting(ax,x,t,data, eps_rel):
    times = data.shape[1]
    
    # eps_colors = [cm.gray(256 - int(128*e/max(eps_rel))) for e in eps_rel]
    
    #  for i in range(len(eps_rel)-1):
    #     ax.fill_between(x[i:i+2],(1,1),(-1,-1), color = eps_colors[i])
   
    ax.plot(x, eps_rel/max(eps_rel)*Ez.max(), color = "k")
    ax.set_ylim((Ez.min()*1.2, Ez.max()*1.2))
    
    ax.plot(x, data[:,0], color = "b")
    for l in range(0,times,30):
        ax.get_lines()[-1].remove()
        ax.plot(x, data[:,l], color = "b")
        plt.pause(0.1)

c = con.speed_of_light
x_span = 20e-6
wavelen = 780e-9
dx = wavelen/50
time_span = 150e-15
source_frequency = c/wavelen*2*np.pi
source_position = 0
source_pulse_length = 100e-15


N = int(x_span/dx)
x_span = N*dx

eps_rel = np.concatenate([np.ones(int(N/6)), 2*np.ones(int(N/3)),np.ones(int(N/3))])

Ez, Hy,x,t = fdtd_1d(eps_rel, dx, time_span,source_frequency, source_position,source_pulse_length)

#%%
fig, ax = plt.subplots(num = 1)
plotting(ax,x,t,Ez, eps_rel)

#%% 

# fig, ax = plt.subplots(num = 0)

# import matplotlib.cm as cm
# eps_colors = [cm.gray(256 - int(128*e/max(eps_rel))) for e in eps_rel]

# for i in range(len(eps_rel)-1):
#     ax.fill_between(x[i:i+2],(1,1),(-1,-1), color = eps_colors[i])