import numpy as np
import scipy.constants as con
import matplotlib.pyplot as plt
import time

#%% Setup
t = 60                    # Total time (s)
dt = 0.01                # Incremental step size (s)
steps = int(t/dt)        # Total simulation steps 

init_speed = 150
angles = np.arange(0,90,5)

rho = 8
A = 0.25e-4
C_d = 0.5
m = 0.05

np.seterr(all='raise') 

def sim(steps, angle):
    bullet_pos = np.zeros((steps,2), dtype = "float64")    # Position of the bullet in x and y (m)
    bullet_speed = np.zeros((steps,2), dtype = "float64")  # Speed of the bullet in x and y (m/s)

    delta = angle/360*2*np.pi
    bullet_speed[0,:] = [np.cos(delta)*init_speed,np.sin(delta)*init_speed]         # Initial bullet speed in x and y (m/s)

    i = 0
    while 1:
        bullet_pos[i+1,0] = bullet_pos[i,0] + bullet_speed[i,0]*dt
        bullet_pos[i+1,1] = bullet_pos[i,1] + bullet_speed[i,1]*dt
        
        bullet_speed[i+1,0] = bullet_speed[i,0] - rho*C_d*A*dt/4/m*bullet_speed[i,0]**2*np.sign(bullet_speed[i,0])
        bullet_speed[i+1,1] = bullet_speed[i,1] - 1/2*con.g*dt - rho*C_d*A*dt/4/m*bullet_speed[i,1]**2*np.sign(bullet_speed[i,1])
        
        if i > steps-3  or bullet_pos[i+1,1] <= 0:
            break
        i += 1
        
    return bullet_pos, bullet_speed, i

#%% Simulation
#
#fig,ax = plt.subplots(num = 0)
#
#t1 = time.time()
#for angle in angles:
#    
#    bullet_pos, bullet_speed = sim(steps, angle)
#    ax.plot(bullet_pos[:,0], bullet_pos[:,1], label = angle, color = 'k')
#    
#print(f"time elapsed: {time.time()-t1} s")
#  
#ax.set_xlim(left = 0)
#ax.set_ylim(bottom = 0)
#ax.set_xlabel("Position (m)")
#ax.set_ylabel("Height (m)")
#ax.legend()

#%% Simulation
    

import seaborn as sns
sns.set(font_scale=1.3)
sns.set_palette("tab20")
sns.set_style("ticks")
sns.set_context("talk")

fig,ax = plt.subplots(num = 1)

t1 = time.time()
angle = 70
aim = 1000
error = 0.1
counter = 0

while counter < 100:
    
    bullet_pos, bullet_speed, index = sim(steps, angle)
    dx = bullet_pos[index, 0] - aim
    
    if np.abs(dx) <  error:
        ax.plot(bullet_pos[:,0], bullet_pos[:,1], label = angle, color = 'red')
        break
    
    angle -= 10*dx/aim
    counter +=1
#    
#    input()1
    
    ax.plot(bullet_pos[:,0], bullet_pos[:,1], label = angle)
#    plt.pause(0.05)
    
print(f"time elapsed: {time.time()-t1} s")
  
ax.set_xlim(left = 0)
ax.set_ylim(bottom = 0)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Height (m)")
#ax.legend()

#%% 
