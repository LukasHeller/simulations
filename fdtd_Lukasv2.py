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



def fdtd_1d(eps_rel, dx, time_span,source_frequency, source_position,source_pulse_length, save_steps):
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
        
    t_save = t[::save_steps]
    N_t_save = len(t_save)
    x = np.arange(0,N_x)*dx
    
    Ez = np.zeros((N_x+1,N_t_save),dtype = eps_rel.dtype)
    Hy = np.zeros((N_x,N_t_save),dtype = eps_rel.dtype)
    
    V_x = c/np.sqrt(eps_rel)
    
    msize = float((Ez.size * Ez.itemsize))*1e-6
    print("Ez and Hy: {:.2f} Mbytes in memory each, dtype {}".format(msize, Ez.dtype))
    
    printProgressBar(0, N_t, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    Ez_temp = np.zeros((N_x+1,2), dtype = eps_rel.dtype)
    Hy_temp = np.zeros((N_x,2), dtype = eps_rel.dtype)
    
    for i in range(N_t-1):
        
        Ez_temp[1:-1,1] = Ez_temp[1:-1,0] + 1/(eps_0*eps_rel[1:])*dt/dx*(Hy_temp[1:,0]-Hy_temp[:-1,0])
        
        curr = 1/(eps_0*eps_rel[100])*dt/dx* \
            np.exp(-((t[i]-3*source_pulse_length)/source_pulse_length)**2)* \
                np.sin(source_frequency*t[i])
                
        Ez_temp[0,1] = Ez_temp[1,0] + (V_x[0]*dt-dx)/(V_x[0]*dt+dx)*(Ez_temp[1,1]-Ez_temp[0,0])
        Ez_temp[-1,1] = Ez_temp[-2,0] + (V_x[-1]*dt-dx)/(V_x[-1]*dt+dx)*(Ez_temp[-2,1]-Ez_temp[-1,0])
        
        Ez_temp[100,1] += curr
        
        Hy_temp[:,1] = Hy_temp[:,0] + 1/mu_0*dt/dx*(Ez_temp[1:,1]-Ez_temp[:-1,1])
        
        if i % save_steps == 0:
            step = divmod(i,save_steps)[0]
            Ez[:,step]= Ez_temp[:,1]
            Hy[:,step]= Hy_temp[:,1]
            printProgressBar(i, N_t, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
        Ez_temp[:,0]= Ez_temp[:,1]
        Hy_temp[:,0]= Hy_temp[:,1]

    Ez = Ez[:-1]
    return Ez, Hy,x,t_save

    
def plotting(ax,x,t, eps_rel, *data, leg = ("1","2")):
    times = data[0].shape[1]
    cols = ("b","r")
    
    for i in range(len(data)):
        data[i][:] = data[i]/data[i].max()
        
    ax.plot(x, eps_rel/max(eps_rel), color = "k")
    ax.set_ylim((-1.2,1.2))
    
    for d,c,l in zip(data, cols, leg):
        ax.plot(x, d[:,0], color = c, label = l)
    ax.legend()
        
    for l in range(0,times,1):
        for delete  in ax.get_lines()[-2:]:
            delete.remove()
        for d,c in zip(data, ["b", "r"]):
            ax.plot(x, d[:,l], color =c)
        plt.pause(0.1)

#%%
c = con.speed_of_light
x_span = 20e-6+106*780e-9/50
wavelen = 780e-9
dx = wavelen/50
time_span = 200e-15
source_frequency = c/wavelen*2*np.pi
source_position = 0
source_pulse_length = 10e-15

save_steps = 60

N = int(x_span/dx)
x_span = N*dx

#%%
eps_rel = np.concatenate([np.ones(int(N/3)), 2*np.ones(int(N/30)),np.ones(int(N/3)),10*np.ones(int(N/30)),np.ones(int(N/3))])

Ez, Hy,x,t = fdtd_1d(eps_rel, dx, time_span,source_frequency, source_position,source_pulse_length, save_steps)

#%%
fig, ax = plt.subplots(num = 0)
plotting(ax,x,t,eps_rel,Ez, Hy, leg = ("E-Field", "H-field"))