import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as con
import matplotlib.cm as cm
from magicMethods import printProgressBar
from pdb import set_trace as trace
from matplotlib.colors import Normalize

#%%
def fdtd_3d(eps_rel, time_span, save_steps, field_component, index, axis, func, dt, dr):

    c = con.speed_of_light
    mu_0 = con.mu_0
    eps_0 = con.epsilon_0
    V_x = c/np.sqrt(eps_rel)
    
    t = np.arange(0,time_span,dt)
    t_save = t[::save_steps]
    
    Nx,Ny,Nz = np.size(eps_rel,0),np.size(eps_rel,1),np.size(eps_rel,2)
    Nt,Nt_save = len(t),len(t_save)

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
            ((Hz_temp[0:Nx,1:Ny,1:Nz,0]-Hz_temp[0:Nx,0:Ny-1,1:Nz,0])- \
             (Hy_temp[0:Nx,1:Ny,1:Nz,0]-Hy_temp[0:Nx,1:Ny,0:Nz-1,0]))
        
        Ey_temp[1:Nx,0:Ny,1:Nz,1] = Ey_temp[1:Nx,0:Ny,1:Nz,0] + \
            1/(eps_0*eps_rel[1:Nx,0:Ny,1:Nz])*dt/dr* \
            ((Hx_temp[1:Nx,0:Ny,1:Nz,0]-Hx_temp[1:Nx,0:Ny,0:Nz-1,0])-(Hz_temp[1:Nx,0:Ny,1:Nz,0]-Hz_temp[0:Nx-1,0:Ny,1:Nz,0]))
        
        Ez_temp[1:Nx,1:Ny,0:Nz,1] = Ez_temp[1:Nx,1:Ny,0:Nz,0] + \
            1/(eps_0*eps_rel[1:Nx,1:Ny,0:Nz])*dt/dr* \
            ((Hy_temp[1:Nx,1:Ny,0:Nz,0]-Hy_temp[0:Nx-1,1:Ny,0:Nz,0])-(Hx_temp[1:Nx,1:Ny,0:Nz,0]-Hx_temp[1:Nx,0:Ny-1,0:Nz,0]))
                    
        Ez_temp[:Nx,:Ny,:Nz,1] += 1/(eps_0*eps_rel[:Nx,:Ny,:Nz])*dt/dr*func(t[time])
        
        Ez_temp[0,1:Ny,1:Nz-1,1] = Ez_temp[1,1:Ny,1:Nz-1,0] + \
            (V_x[0,1:Ny,1:Nz-1]*dt-dr)/(V_x[0,1:Ny,1:Nz-1]*dt+dr)* \
            (Ez_temp[1,1:Ny,1:Nz-1,1]-Ez_temp[0,1:Ny,1:Nz-1,0])
        
        Ez_temp[-1,1:Ny,1:Nz-1,1] = Ez_temp[-2,1:Ny,1:Nz-1,0] + (V_x[-1,1:Ny,1:Nz-1]*dt-dr)/(V_x[-1,1:Ny,1:Nz-1]*dt+dr)*(Ez_temp[-2,1:Ny,1:Nz-1,1]-Ez_temp[-1,1:Ny,1:Nz-1,0])
        Ey_temp[0,1:Ny-1,1:Nz,1] = Ey_temp[1,1:Ny-1,1:Nz,0] + (V_x[0,1:Ny-1,1:Nz]*dt-dr)/(V_x[0,1:Ny-1,1:Nz]*dt+dr)*(Ey_temp[1,1:Ny-1,1:Nz,1]-Ey_temp[0,1:Ny-1,1:Nz,0])
        Ey_temp[-1,1:Ny-1,1:Nz,1] = Ey_temp[-2,1:Ny-1,1:Nz,0] + (V_x[-1,1:Ny-1,1:Nz]*dt-dr)/(V_x[-1,1:Ny-1,1:Nz]*dt+dr)*(Ey_temp[-2,1:Ny-1,1:Nz,1]-Ey_temp[-1,1:Ny-1,1:Nz,0])
        
        
        Ez_temp[1:Nx,0,1:Nz-1,1] = Ez_temp[1:Nx,1,1:Nz-1,0] + (V_x[1:Nx,0,1:Nz-1]*dt-dr)/(V_x[1:Nx,0,1:Nz-1]*dt+dr)*(Ez_temp[1:Nx,1,1:Nz-1,1]-Ez_temp[1:Nx,0,1:Nz-1,0])
        Ez_temp[1:Nx,-1,1:Nz-1,1] = Ez_temp[1:Nx,-2,1:Nz-1,0] + (V_x[1:Nx,-1,1:Nz-1]*dt-dr)/(V_x[1:Nx,-1,1:Nz-1]*dt+dr)*(Ez_temp[1:Nx,-2,1:Nz-1,1]-Ez_temp[1:Nx,-1,1:Nz-1,0])
        Ex_temp[1:Nx-1,0,1:Nz,1] = Ex_temp[1:Nx-1,1,1:Nz,0] + (V_x[1:Nx-1,0,1:Nz]*dt-dr)/(V_x[1:Nx-1,0,1:Nz]*dt+dr)*(Ex_temp[1:Nx-1,1,1:Nz,1]-Ex_temp[1:Nx-1,0,1:Nz,0])
        Ex_temp[1:Nx-1,-1,1:Nz,1] = Ex_temp[1:Nx-1,-2,1:Nz,0] + (V_x[1:Nx-1,-1,1:Nz]*dt-dr)/(V_x[1:Nx-1,-1,1:Nz]*dt+dr)*(Ex_temp[1:Nx-1,-2,1:Nz,1]-Ex_temp[1:Nx-1,-1,1:Nz,0])

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
    ax.set_aspect('equal', adjustable='box')
    
    for l in range(0,times,1):
        ax.clear()
        im = ax.pcolormesh(x/xyaxis_f[1], y/xyaxis_f[1], data[:,:,l], label = leg, cmap = cmap, norm = Normalize(-1,1))
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
            frequency = c/wavelen*2*np.pi
            
            movex, movez = 75,75
            # R = offset*(1+(zr/offset)**2)
            jz = np.exp(-((x-movex*dr)**2 + (z-movez*dr)**2)/w**2)
            jz[:,3:,:] = 0
            # phasefactor = k*((x-movex*dr)**2 + (z-movez*dr)**2)/2/R + k*offset - np.arctan(offset/zr)
            # return np.exp(-((t-2*pulse_length)/pulse_length)**2)* \
            #        np.sin(frequency*t)*jz*np.sin(phasefactor)
            return np.exp(-((t-2*pulse_length)/pulse_length)**2)* \
                   np.sin(frequency*t)*jz
                       
        return pulse_wrapper
    
    @classmethod
    def pullOn_gauss_pulse(self,frequency,switch_on,w_0,x,y,z,movex, movez):
        def pulse_wrapper(t):
            jz = np.exp(-((x-movex)**2 + (z-movez)**2)/w_0**2)
            jz[:,3:,:] = 0
            return (np.arctan((x/pulse_length - 1)*25)/np.pi+0.5)* \
                   np.sin(frequency*t)*jz
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
    
class EpsLandscape():
    def __init__(self,eps_rel_b, x_dim, y_dim, z_dim, dr):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.dr = dr
        
        self.Nx, self.Ny, self.Nz = int(x_dim/dr),int(y_dim/dr),int(z_dim/dr)
        self.x,self.y,self.z = np.meshgrid(np.arange(0,self.Nx)*dr,
                                           np.arange(0,self.Ny)*dr,
                                           np.arange(self.Nz)*dr, indexing = 'ij')

        
        self.eps_rel = eps_rel_b* \
            np.ones([self.Nx, self.Ny, self.Nz], dtype = "float32")
    
    def block(self, eps_rel,x,y,z):
        x1,x2 = [int(i/self.dr) for i in x]
        y1,y2 = [int(i/self.dr) for i in y]
        z1,z2 = [int(i/self.dr) for i in z]
        
        self.eps_rel[x1:x2, y1:y2, z1:z2] = eps_rel
    
    def show_epsLandscape(self,num):
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, num = num)
        cmap = plt.get_cmap('gray')
        for a in (ax1,ax2,ax3):
            a.set_aspect('equal', adjustable='box')
        
        f = 1e-6
        f_s = r"$\mu$m"
        n =5
        
        order, order_s = (self.y,self.z), ("Y","Z")
        if self.Ny > self.Nz:
            order,order_s = (self.z,self.y),("Z","Y")
        ax1.set_xlabel(f"{order_s[0]} ({f_s})")
        ax1.set_ylabel(f"{order_s[1]} ({f_s})")
        for i in range(0,self.Nx, int(self.Nx/5)):
            ax1.pcolormesh(order[0][i,::n,::n]/f,order[1][i,::n,::n]/f,self.eps_rel[i,::n,::n], 
                           norm = Normalize(), alpha = 0.1,
                           shading='gouraud')
            
        order, order_s = (self.z,self.x), ("Z","X")
        if self.Nz > self.Nx:
            order,order_s = (self.x,self.z),("X","Z")
        ax2.set_xlabel(f"{order_s[0]} ({f_s})")
        ax2.set_ylabel(f"{order_s[1]} ({f_s})")
        for i in range(0,self.Ny, int(self.Ny/5)):
            ax2.pcolormesh(order[0][::n,i,::n]/f,order[1][::n,i,::n]/f,self.eps_rel[::n,i,::n], 
                           norm = Normalize(), alpha = 0.1,
                           shading='gouraud')
            
        order, order_s = (self.x,self.y), ("X","Y")
        if self.Nx > self.Ny:
            order,order_s = (self.y,self.x),("Y","X")
        ax3.set_xlabel(f"{order_s[0]} ({f_s})")
        ax3.set_ylabel(f"{order_s[1]} ({f_s})")
        for i in range(0,self.Nz, int(self.Nz/5)):
            ax3.pcolormesh(order[0][::n,::n,i]/f,order[1][::n,::n,i]/f,self.eps_rel[::n,::n,i], 
                           norm = Normalize(), alpha = 0.1,
                           shading='gouraud')
        plt.tight_layout()
        
#%% Setup
c = con.speed_of_light
wavelen = 780e-9
frequency = c/wavelen*2*np.pi
dr = wavelen/25
dt = dr/2/c
time_span = 100e-15
save_steps = 8

source_width = 25*dr
pulse_length = 20e-15
movex, movez = 3e-6, 3e-6
    
#%%
eps = EpsLandscape(1, 6e-6, 18e-6, 6e-6, dr)
eps.block(2, (0,6e-6), (2e-6,14e-6), (0,6e-6))
eps.block(3, (1.25e-6,4.75e-6), (2e-6,14e-6), (1.25e-6,4.75e-6))
eps.show_epsLandscape(3)
 
#%%
func = Sources.pullOn_gauss_pulse(frequency, pulse_length,source_width,eps.x,eps.y,eps.z,movex,movez)
x, y, data, t = fdtd_3d(eps.eps_rel, time_span, save_steps, "Hx", int(eps.Nz/2), "z", func, dt, dr)


#%% Plotting
fig, ax = plt.subplots(num = 0)      
plotting(ax, x, y, data[:,:,:], t, leg = "label")

# #%%
# fig, ax = plt.subplots(num = 0)   
# def filming(ax, x, y, data, t):
    
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.animation as manimation
    
#     FFMpegWriter = manimation.writers['ffmpeg']
#     metadata = dict(title='Movie Test', artist='Matplotlib',
#                     comment='Movie support!')
#     writer = FFMpegWriter(fps=10, metadata=metadata)

#     times = data.shape[-1]
#     time_f = ("fs", 1e-15)
#     xyaxis_f = ("$\mu$m", 1e-6)
    
#     data = data/data.max()
#     cmap = plt.get_cmap('bwr')
    
#     from matplotlib.colors import Normalize
#     ax.pcolormesh(x/xyaxis_f[1], y/xyaxis_f[1], data[:,:,0],cmap = cmap, norm = Normalize(-1,1))
#     ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[0]/time_f[1],time_f[0]), transform=ax.transAxes)
    
#     with writer.saving(fig, "writer_test_2.mp4", 100):
#         for l in range(0,times,1):
#             ax.clear()
#             ax.pcolormesh(x/xyaxis_f[1], y/xyaxis_f[1], data[:,:,l], cmap = cmap, norm = Normalize(-1,1))
#             ax.text(0.8, 0.2,"t = {:.2f} {}".format(t[l]/time_f[1],time_f[0]), transform=ax.transAxes)
#             ax.set_aspect('equal', adjustable='box')
#             plt.pause(0.1)
#             print(l)
#             writer.grab_frame()
# filming(ax, x, y, data[:,:,:], t)
# import matplotlib
# matplotlib.use("Qt5Agg")


