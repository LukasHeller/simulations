import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con
import matplotlib.cm as cm
from magicMethods import printProgressBar

#%%
def fdtd_1d(eps_rel, time_span, save_steps, func, dt, dx):
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
    V_x = c/np.sqrt(eps_rel)
    
    t = np.arange(0,time_span,dt)
    t_save = t[::save_steps]
    
    N_x, N_t, N_t_save = len(eps_rel), len(t), len(t_save)

    x = np.arange(0,N_x)*dx
    
    Ez = np.zeros((N_x+1,N_t_save),dtype = eps_rel.dtype)
    Hy = np.zeros((N_x,N_t_save),dtype = eps_rel.dtype)
    Ez_temp = np.zeros((N_x+1,2), dtype = eps_rel.dtype)
    Hy_temp = np.zeros((N_x,2), dtype = eps_rel.dtype)
    
    print("Ez and Hy: {:.2f} Mbytes in memory each, dtype {}"
          .format(float((Ez.size * Ez.itemsize))*1e-6, Ez.dtype))
    
    printProgressBar(0, N_t, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i in range(N_t-1):
       
        Ez_temp[1:-1,1] = Ez_temp[1:-1,0] + 1/(eps_0*eps_rel[1:])*dt/dx*(Hy_temp[1:,0]-Hy_temp[:-1,0])
                
        Ez_temp[0,1] = Ez_temp[1,0] + (V_x[0]*dt-dx)/(V_x[0]*dt+dx)*(Ez_temp[1,1]-Ez_temp[0,0])
        Ez_temp[-1,1] = Ez_temp[-2,0] + (V_x[-1]*dt-dx)/(V_x[-1]*dt+dx)*(Ez_temp[-2,1]-Ez_temp[-1,0])
        
        Ez_temp[func.index_position,1] += 1/(eps_0*eps_rel[func.index_position])*dt/dx*func(t[i])
        
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
    time_f = ("fs", 1e-15)
    xaxis_f = ("$\mu$m", 1e-6)
    cols = ("b","r")
    
    for i in range(len(data)):
        data[i][:] = data[i]/data[i].max()
        
    ax.plot(x/xaxis_f[1], eps_rel/max(eps_rel), color = "k")
    ax.set_ylim((-1.2,1.2))
    
    for d,c,l in zip(data, cols, leg):
        ax.plot(x/xaxis_f[1], d[:,0], color = c, label = l)
    ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[0]/time_f[1],time_f[0]), transform=ax.transAxes)
    
    ax.legend()
    ax.set_xlabel(f"Distance ({xaxis_f[0]})")    
    
    for l in range(0,times,1):
        for delete  in ax.get_lines()[-2:]:
            delete.remove()
        ax.texts[0].remove()
        for d,c in zip(data, ["b", "r"]):
            ax.plot(x/xaxis_f[1], d[:,l], color =c)
        ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[l]/time_f[1],time_f[0]), transform=ax.transAxes)
        plt.pause(0.1)
        
#%% class with methods for different source terms
class Sources():
    
    c = con.speed_of_light
    mu_0 = con.mu_0
    eps_0 = con.epsilon_0
    
    @classmethod
    def gaussian_pulse(self,frequency,index_position,pulse_length):
        def pulse_wrapper(t):
            return np.exp(-((t-3*pulse_length)/pulse_length)**2)* \
                   np.sin(frequency*t)
        pulse_wrapper.index_position = index_position
        return pulse_wrapper
    
    @classmethod
    def delta_pulse(self,index_position, dt, index_time):
        def pulse_wrapper(t):
            if t > dt*index_time  and t < dt*(index_time + 2)  :
                return 100
            else:
                return 0
        pulse_wrapper.index_position = index_position
        return pulse_wrapper

#%% Setup
c = con.speed_of_light
wavelen = 780e-9
dx = wavelen/10
dt = dx/2/c
time_span = 300e-15
frequency = c/wavelen*2*np.pi
index_position = 100
pulse_length = 10e-15

save_steps = 10

x_span = 20e-6+106*780e-9/50
N = int(x_span/dx)
    
#%% Simulation
eps_rel = np.concatenate([np.ones(int(N/3)), 4*np.ones(int(N/30)),np.ones(int(N/3)),10*np.ones(int(N/30)),np.ones(int(N/3))])

# eps_rel = np.concatenate([np.ones(int(N))])
func = Sources.gaussian_pulse(frequency,index_position,pulse_length)
# func = Sources.delta_pulse(index_position,dt, 100)
Ez, Hy,x,t = fdtd_1d(eps_rel, time_span, save_steps, func, dt, dx)

#%% Plotting
fig, ax = plt.subplots(num = 0)
plotting(ax,x,t,eps_rel,Ez, Hy, leg = ("E-Field", "H-field"))