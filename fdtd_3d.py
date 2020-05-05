import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con
import matplotlib.cm as cm
from magicMethods import printProgressBar
from pdb import set_trace as trace

#%%
def fdtd_3d(eps_rel, time_span, save_steps, field_component, index, axis, func, dt, dr):
    """
    field_component,z_index,output_step)
    function [F, t] = fdtd_3d(eps_rel,dr,time_span,freq,tau,jx,jy,jz,...
     field_component,z_index,output_step)
    Computes the temporal evolution of a pulsed spatially extended current source using
    the 3D FDTD method. Returns xy-slices of the select field at the given z-
    position every output_step time steps. The pulse is centered at a simulation time
    of 3*tau. All quantities have to be specified in SI units.
    Arguments:
     eps_rel
     : rel.permittivity distribution within the computational domain
     (3D array)
     dr
     : grid spacing (scalar, please ensure dr<=lambda/20)
     time_span
     : time span of simulation (scalar)
     freq
     : center frequency of the current source (scalar)
     tau
     : temporal width of Gaussian envelope of the source (scalar)
     jx,jy,jz
     : spatial density profile of current source (3D arrays)
     field_component : field component which is stored
     (one of ‘ex’,’ey’,’ez’,’hx’,’hy’,’hz’)
     output_step
     : number of time steps between field outputs (integer)
    Returns:
     F : xy-slices of the selected field component every output_step time steps
     (3D array, time varies along the last dimension)
     t : time of the field output (vector)

    """
    
    c = con.speed_of_light
    mu_0 = con.mu_0
    eps_0 = con.epsilon_0
    V_x = c/np.sqrt(eps_rel)
    
    t = np.arange(0,time_span,dt)
    t_save = t[::save_steps]
    
    Nx,Ny,Nz = np.size(eps_rel,0),np.size(eps_rel,1),np.size(eps_rel,2)
    Nt,Nt_save = len(t),len(t_save)
    
    # source_index = func.index_position + (1,)
    # print(source_index)
    
    if axis == "z":
        d1 = np.arange(0,Nx)*dr
        d2 = np.arange(0,Ny)*dr
        out = np.zeros((Nx,Ny,Nt_save),dtype = eps_rel.dtype)
    elif axis == "x":
        d1 = np.arange(0,Ny)*dr
        d2 = np.arange(0,Nz)*dr
        out = np.zeros((Ny,Nz,Nt_save),dtype = eps_rel.dtype)
    elif axis == "y":
        d1 = np.arange(0,Nz)*dr
        d2 = np.arange(0,Nx)*dr
        out = np.zeros((Nz,Nx,Nt_save),dtype = eps_rel.dtype)
    else:
        raise ValueError("'axis' has to be either 'x','y' or 'z'")
    x,y = np.meshgrid(d1,d2,indexing = 'ij')
    
    Ex_temp = np.zeros((Nx,Ny+1,Nz+1,2),dtype = eps_rel.dtype)
    Ey_temp = np.zeros((Nx+1,Ny,Nz+1,2),dtype = eps_rel.dtype)
    Ez_temp = np.zeros((Nx+1,Ny+1,Nz,2),dtype = eps_rel.dtype)
    Hx_temp = np.zeros((Nx+1,Ny,Nz,2),dtype = eps_rel.dtype)
    Hy_temp = np.zeros((Nx,Ny+1,Nz,2),dtype = eps_rel.dtype)
    Hz_temp = np.zeros((Nx,Ny,Nz+1,2),dtype = eps_rel.dtype) 
    
    print("Field component '{}' along axis '{}': {:.2f} Mbytes in memory, dtype: {}"
           .format(field_component,axis, float((out.size * out.itemsize))*1e-6, eps_rel.dtype))
    print("Total RAM ocupied during calculation: {:.2f} Mbytes in memory"
           .format(float(out.size * out.itemsize + \
                         6*Ex_temp.size * Ex_temp.itemsize + \
                         2*x.size*x.itemsize)*1e-6))
    
    printProgressBar(0, Nt, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for time in range(Nt-1):
        Ex_temp[0:Nx,1:Ny,1:Nz,1] = Ex_temp[0:Nx,1:Ny,1:Nz,0] + \
            1/(eps_0*eps_rel[0:Nx,1:Ny,1:Nz])*dt/dr* \
            ((Hz_temp[0:Nx,1:Ny,1:Nz,0]-Hz_temp[0:Nx,0:Ny-1,1:Nz,0])-(Hy_temp[0:Nx,1:Ny,1:Nz,0]-Hy_temp[0:Nx,1:Ny,0:Nz-1,0]))
        
        Ey_temp[1:Nx,0:Ny,1:Nz,1] = Ey_temp[1:Nx,0:Ny,1:Nz,0] + \
            1/(eps_0*eps_rel[1:Nx,0:Ny,1:Nz])*dt/dr* \
            ((Hx_temp[1:Nx,0:Ny,1:Nz,0]-Hx_temp[1:Nx,0:Ny,0:Nz-1,0])-(Hz_temp[1:Nx,0:Ny,1:Nz,0]-Hz_temp[0:Nx-1,0:Ny,1:Nz,0]))
        
        Ez_temp[1:Nx,1:Ny,0:Nz,1] = Ez_temp[1:Nx,1:Ny,0:Nz,0] + \
            1/(eps_0*eps_rel[1:Nx,1:Ny,0:Nz])*dt/dr* \
            ((Hy_temp[1:Nx,1:Ny,0:Nz,0]-Hy_temp[0:Nx-1,1:Ny,0:Nz,0])-(Hx_temp[1:Nx,1:Ny,0:Nz,0]-Hx_temp[1:Nx,0:Ny-1,0:Nz,0]))
                    
        # Ez_temp[source_index] += 1/(eps_0*eps_rel[func.index_position])*dt/dr*func(t[i])
        Ez_temp[:Nx,:Ny,:Nz,1] += 1/(eps_0*eps_rel[:Nx,:Ny,:Nz])*dt/dr*func(t[time])
        
        # Ez_temp[32,33,6,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])
        # Ez_temp[33,32,6,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])
        # Ez_temp[33,33,6,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])

        # Ez_temp[32,32,5,1] += 1/(eps_0*eps_rel[32,32,1])*dt/dr*func(t[time])
        # Ez_temp[32,33,5,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])
        # Ez_temp[33,32,5,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])
        # Ez_temp[33,33,5,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[time])
        
        # Ez_temp[32,32,7,1] += 1/(eps_0*eps_rel[32,32,1])*dt/dr*func(t[i])
        # Ez_temp[32,33,7,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[i])
        # Ez_temp[33,32,7,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[i])
        # Ez_temp[33,33,7,1] += 1/(eps_0*eps_rel[32,33,1])*dt/dr*func(t[i])
        
        
#---------------------------------
        # trace()
    # Ex_temp = np.zeros((Nx,Ny+1,Nz+1,2),dtype = eps_rel.dtype)
    # Ey_temp = np.zeros((Nx+1,Ny,Nz+1,2),dtype = eps_rel.dtype)
    # Ez_temp = np.zeros((Nx+1,Ny+1,Nz,2),dtype = eps_rel.dtype)
        # Ez_temp[0,0:Ny+1,0:Nz,1] = Ez_temp[1,0:Ny+1,0:Nz,0] + (c*dt-dr)/(c*dt+dr)*(Ez_temp[1,0:Ny+1,0:Nz,1]-Ez_temp[0,0:Ny+1,0:Nz,0])
        # Ez_temp[-1,0:Ny+1,0:Nz,1] = Ez_temp[-2,0:Ny+1,0:Nz,0] + (c*dt-dr)/(c*dt+dr)*(Ez_temp[-2,0:Ny+1,0:Nz,1]-Ez_temp[-1,0:Ny+1,0:Nz,0])
        # Ey_temp[0,0:Ny,0:Nz+1,1] = Ey_temp[1,0:Ny,0:Nz+1,0] + (c*dt-dr)/(c*dt+dr)*(Ey_temp[1,0:Ny,0:Nz+1,1]-Ey_temp[0,0:Ny,0:Nz+1,0])
        # Ey_temp[-1,0:Ny,0:Nz+1,1] = Ey_temp[-2,0:Ny,0:Nz+1,0] + (c*dt-dr)/(c*dt+dr)*(Ey_temp[-2,0:Ny,0:Nz+1,1]-Ey_temp[-1,0:Ny,0:Nz+1,0])
        
        
        # Ez_temp[0:Nx+1,0,0:Nz,1] = Ez_temp[0:Nx+1,1,0:Nz,0] + (c*dt-dr)/(c*dt+dr)*(Ez_temp[0:Nx+1,1,0:Nz,1]-Ez_temp[0:Nx+1,0,0:Nz,0])
        # Ez_temp[0:Nx+1,-1,0:Nz,1] = Ez_temp[0:Nx+1,-2,0:Nz,0] + (c*dt-dr)/(c*dt+dr)*(Ez_temp[0:Nx+1,-2,0:Nz,1]-Ez_temp[0:Nx+1,-1,0:Nz,0])
        # Ex_temp[0:Nx,0,0:Nz+1,1] = Ex_temp[0:Nx,1,0:Nz+1,0] + (c*dt-dr)/(c*dt+dr)*(Ex_temp[0:Nx,1,0:Nz+1,1]-Ex_temp[0:Nx,0,0:Nz+1,0])
        # Ex_temp[0:Nx,-1,0:Nz+1,1] = Ex_temp[0:Nx,-2,0:Nz+1,0] + (c*dt-dr)/(c*dt+dr)*(Ex_temp[0:Nx,-2,0:Nz+1,1]-Ex_temp[0:Nx,-1,0:Nz+1,0])
        
        
        Ez_temp[0,1:Ny,1:Nz-1,1] = Ez_temp[1,1:Ny,1:Nz-1,0] + (V_x[0,1:Ny,1:Nz-1]*dt-dr)/(V_x[0,1:Ny,1:Nz-1]*dt+dr)*(Ez_temp[1,1:Ny,1:Nz-1,1]-Ez_temp[0,1:Ny,1:Nz-1,0])
        Ez_temp[-1,1:Ny,1:Nz-1,1] = Ez_temp[-2,1:Ny,1:Nz-1,0] + (V_x[-1,1:Ny,1:Nz-1]*dt-dr)/(V_x[-1,1:Ny,1:Nz-1]*dt+dr)*(Ez_temp[-2,1:Ny,1:Nz-1,1]-Ez_temp[-1,1:Ny,1:Nz-1,0])
        Ey_temp[0,1:Ny-1,1:Nz,1] = Ey_temp[1,1:Ny-1,1:Nz,0] + (V_x[0,1:Ny-1,1:Nz]*dt-dr)/(V_x[0,1:Ny-1,1:Nz]*dt+dr)*(Ey_temp[1,1:Ny-1,1:Nz,1]-Ey_temp[0,1:Ny-1,1:Nz,0])
        Ey_temp[-1,1:Ny-1,1:Nz,1] = Ey_temp[-2,1:Ny-1,1:Nz,0] + (V_x[-1,1:Ny-1,1:Nz]*dt-dr)/(V_x[-1,1:Ny-1,1:Nz]*dt+dr)*(Ey_temp[-2,1:Ny-1,1:Nz,1]-Ey_temp[-1,1:Ny-1,1:Nz,0])
        
        
        Ez_temp[1:Nx,0,1:Nz-1,1] = Ez_temp[1:Nx,1,1:Nz-1,0] + (V_x[1:Nx,0,1:Nz-1]*dt-dr)/(V_x[1:Nx,0,1:Nz-1]*dt+dr)*(Ez_temp[1:Nx,1,1:Nz-1,1]-Ez_temp[1:Nx,0,1:Nz-1,0])
        Ez_temp[1:Nx,-1,1:Nz-1,1] = Ez_temp[1:Nx,-2,1:Nz-1,0] + (V_x[1:Nx,-1,1:Nz-1]*dt-dr)/(V_x[1:Nx,-1,1:Nz-1]*dt+dr)*(Ez_temp[1:Nx,-2,1:Nz-1,1]-Ez_temp[1:Nx,-1,1:Nz-1,0])
        Ex_temp[1:Nx-1,0,1:Nz,1] = Ex_temp[1:Nx-1,1,1:Nz,0] + (V_x[1:Nx-1,0,1:Nz]*dt-dr)/(V_x[1:Nx-1,0,1:Nz]*dt+dr)*(Ex_temp[1:Nx-1,1,1:Nz,1]-Ex_temp[1:Nx-1,0,1:Nz,0])
        Ex_temp[1:Nx-1,-1,1:Nz,1] = Ex_temp[1:Nx-1,-2,1:Nz,0] + (V_x[1:Nx-1,-1,1:Nz]*dt-dr)/(V_x[1:Nx-1,-1,1:Nz]*dt+dr)*(Ex_temp[1:Nx-1,-2,1:Nz,1]-Ex_temp[1:Nx-1,-1,1:Nz,0])
     
        
        # Ez_temp[-1,i+1] = Ez[-2,i] + (V_x[-1]*dt-dx)/(V_x[-1]*dt+dx)*(Ez[-2,i+1]-Ez[-1,i])
        # Ez[0,i+1] = Ez[1,i] + (V_x[0]*dt-dx)/(V_x[0]*dt+dx)*(Ez[1,i+1]-Ez[0,i])
        # Ez[-1,i+1] = Ez[-2,i] + (V_x[-1]*dt-dx)/(V_x[-1]*dt+dx)*(Ez[-2,i+1]-Ez[-1,i])

#---------------------------------





        Hx_temp[1:Nx,0:Ny,0:Nz,1] = Hx_temp[1:Nx,0:Ny,0:Nz,0] + \
            1/mu_0*dt/dr* \
            ((Ey_temp[1:Nx,0:Ny,1:Nz+1,1]-Ey_temp[1:Nx,0:Ny,0:Nz,1])-(Ez_temp[1:Nx,1:Ny+1,0:Nz,1]-Ez_temp[1:Nx,0:Ny,0:Nz,1]))
        Hy_temp[0:Nx,1:Ny,0:Nz,1] = Hy_temp[0:Nx,1:Ny,0:Nz,0] + \
            1/mu_0*dt/dr* \
            ((Ez_temp[1:Nx+1,1:Ny,0:Nz,1]-Ez_temp[0:Nx,1:Ny,0:Nz,1])-(Ex_temp[0:Nx,1:Ny,1:Nz+1,1]-Ex_temp[0:Nx,1:Ny,0:Nz,1]))
        Hz_temp[0:Nx,0:Ny,1:Nz,1] = Hz_temp[0:Nx,0:Ny,1:Nz,0] + \
            1/mu_0*dt/dr* \
            ((Ex_temp[0:Nx,1:Ny+1,1:Nz,1]-Ex_temp[0:Nx,0:Ny,1:Nz,1])-(Ey_temp[1:Nx+1,0:Ny,1:Nz,1]-Ey_temp[0:Nx,0:Ny,1:Nz,1]))
           
        if time % save_steps == 0:
            step = divmod(time,save_steps)[0]
            out[:,:,step] = Ez_temp[:Nx,:Ny,index,1]
            printProgressBar(time, Nt, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
        Ex_temp[:,:,:,0]= Ex_temp[:,:,:,1]
        Ey_temp[:,:,:,0]= Ey_temp[:,:,:,1]
        Ez_temp[:,:,:,0]= Ez_temp[:,:,:,1]
        Hx_temp[:,:,:,0]= Hx_temp[:,:,:,1]
        Hy_temp[:,:,:,0]= Hy_temp[:,:,:,1]
        Hz_temp[:,:,:,0]= Hz_temp[:,:,:,1]
           
    return x, y , out, t_save
    
def plotting(ax, x, y, data, t, leg = "label"):
    times = data.shape[-1]
    time_f = ("fs", 1e-15)
    xyaxis_f = ("$\mu$m", 1e-6)
    
    data = data/data.max()
    cmap = plt.get_cmap('bwr')
    
    from matplotlib.colors import Normalize
    ax.pcolormesh(x/xyaxis_f[1], y/xyaxis_f[1], data[:,:,0], label = leg, cmap = cmap, norm = Normalize(-1,1))
    ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[0]/time_f[1],time_f[0]), transform=ax.transAxes)
    
    # ax.set_xlabel(f"Distance ({xaxis_f[0]})")    
    
    for l in range(0,times,1):
        ax.clear()
        im = ax.pcolormesh(x/xyaxis_f[1], y/xyaxis_f[1], data[:,:,l], label = leg, cmap = cmap, norm = Normalize(-1,1))
        # fig.colorbar(im)
        ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[l]/time_f[1],time_f[0]), transform=ax.transAxes)
        plt.pause(0.4)
        print(l)
        
#%% class with methods for different source terms
class Sources():
    
    c = con.speed_of_light
    mu_0 = con.mu_0
    eps_0 = con.epsilon_0
    
    
    @classmethod
    def gaussian_pulse(self,frequency,index_position,pulse_length, w_0, wavelen,x,y,z,dr):
        def pulse_wrapper(t):
            
            offset = 200*dr
            zr = np.pi*w_0**2/wavelen
            w = w_0*np.sqrt(1+(offset/zr)**2)
            k = 2*np.pi/wavelen
            
            movex, movez = 75,75
            R = offset*(1+(zr/offset)**2)
            jz = np.exp(-((x-movex*dr)**2 + (z-movez*dr)**2)/w**2)
            jz[:,3:,:] = 0
            phasefactor = k*((x-movex*dr)**2 + (z-movez*dr)**2)/2/R + k*offset - np.arctan(offset/zr)
            # return np.exp(-((t-2*pulse_length)/pulse_length)**2)* \
            #        np.sin(frequency*t)*jz*np.sin(phasefactor)
            return np.sin(frequency*t)*jz*np.sin(phasefactor)
                       
        return pulse_wrapper
    
    @classmethod
    def pullOn_gauss_pulse(self,frequency,switch_on,w_0,x,y,z,dr):
        def pulse_wrapper(t):
            
            movex, movez = 75,75
            jz = np.exp(-((x-movex*dr)**2 + (z-movez*dr)**2)/w_0**2)
            jz[:,3:,:] = 0
            return (np.arctan((x/pulse_length - 1)*25)/np.pi+0.5)* \
                   np.sin(frequency*t)*jz

        return pulse_wrapper
    
    #     @classmethod
    # def gaussian_pulse(self,frequency,index_position,pulse_length,jz):
    #     def pulse_wrapper(t):
    #         return np.exp(-((t-3*pulse_length)/pulse_length)**2)* \
    #                np.sin(frequency*t)
    #     pulse_wrapper.index_position = index_position
    #     pulse_wrapper.jz = jz
    #     print(jz)
    #     return pulse_wrapper
    
    @classmethod
    def delta_pulse(self,index_position, dt, index_time):
        def pulse_wrapper(t):
            if t > dt*index_time  and t < dt*(index_time + 2)  :
                return 100
            else:
                return 0
        pulse_wrapper.index_position = index_position
        return pulse_wrapper
    
class EpsLandscape():
    def __init__(self,eps_rel_b, x_dim, y_dim, z_dim, dr):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.dr = dr
        
        
        
        self.eps_rel = eps_rel_b* \
            np.ones([int(x_dim/dr),int(y_dim/dr),int(z_dim/dr)], dtype = "float32")
    
    def block(self, eps_rel,x,y,z):
        x1,x2 = [int(i/self.dr) for i in x]
        y1,y2 = [int(i/self.dr) for i in y]
        z1,z2 = [int(i/self.dr) for i in z]
        
        self.eps_rel[x1:x2, y1:y2, z1:z2] = eps_rel
    
    def show_epsLandscape():
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.pcolormesh
#%% Setup
c = con.speed_of_light
wavelen = 780e-9
dr = wavelen/25
dt = dr/2/c
time_span = 5e-15

frequency = c/wavelen*2*np.pi
source_width = 30*dr
pulse_length = 20e-15

save_steps = 8

x_span = 5e-6
N = int(x_span/dr)

Nx, Ny, Nz = 130,250,130
    
#%% Simulation
eps_rel = np.ones([Nx,Ny,Nz], dtype = "float32")

x,y,z = np.meshgrid(np.arange(0,Nx)*dr,np.arange(0,Ny)*dr,np.arange(Nz), indexing = 'ij')

func = Sources.gaussian_pulse(frequency,index_position,pulse_length, source_width, wavelen,x,y,z,dr)
x, y, data, t = fdtd_3d(eps_rel, time_span, save_steps, "Hx", 1, "z", func, dt, dr)

#%% Plotting
fig, ax = plt.subplots(num = 0)      
plotting(ax, x, y, data[:,:,:], t, leg = "label")



#%%
# r,z = np.meshgrid(np.linspace(-6,6,300)*800e-9,np.linspace(-10,10,300)*800e-9, indexing = 'ij')
# dr = wavelen/25
# w_0 = wavelen*1.5
# zr = np.pi*w_0**2/wavelen
# w = w_0*np.sqrt(1+(z/zr)**2)
# k = 2*np.pi/wavelen
            
# R = z*(1+(zr/z)**2)
# jz = np.exp(-((r)**2)/w**2)

# phasefactor = k*((r)**2)/2/R + k*z - np.arctan(z/zr)
# plt.figure(4)
# plt.pcolormesh(z,r,(jz*np.sin(phasefactor))**2)




