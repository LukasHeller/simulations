import numpy as np
import matplotlib.pyplot as plt

# --- Calculating efficiency based on assumption that ---
# --- there can be as many photons per write/read gate---

# No noise, no dephasing is taken into account here

p = np.arange(0.01,1,0.01) # p ranging from 0 to 1
eta_det = 0.05             # Detection efficiency (same for write and read here)
eta_read = 0.5             # Readout efficiency
pw_geo = eta_det*(1/(1-p)-1)    #pw, based on geometric series
pwr_geo = eta_det**2*(1/(1-p)-1)*eta_read # pwr, based on geometric series

# --- Calculating efficiency based on assumption that ---
# --- only one photon can be detected per write/read gate ---

pw = np.zeros(len(p))
pwr = np.zeros(len(p))

for i,x in enumerate(p):
    print(f'\nprobability: {x}')
    pw_list = np.zeros(100)
    pwr_list = np.zeros(100)
    for n in range(len(pw_list)):
        pw_list[n] = eta_det*x**(n+1)*(1-np.sum(pw_list))
        pwr_list[n] = eta_det**2*x**(n+1)*eta_read*(1-np.sum(pwr_list))
        print(f'n: {n}')
        print(f'\tpw_list[n]:{pw_list[n]}, \tpr_list[n] {pwr_list[n]}')
    
    pw[i] = np.sum(pw_list)
    pwr[i] = np.sum(pwr_list)
    print(pw[i])

prw = pwr/pw

#%% --- Plotting
plt.figure(0)
plt.plot(p,pw*100, label = r'$p_w$')
plt.plot(p,pwr*100, label = r'$p_{w,r}$')
plt.plot(p,prw*100, label = r'$p_{r|w}$')
plt.plot(p,pw_geo*100, label = r'$p_{w}$ (geo)')
plt.plot(p,pwr_geo*100, label = r'$p_{w,r}$ (geo)')
plt.ylabel(r'p [%]')
plt.xlabel(r'$p$')
plt.ylim([0,110])
plt.xlim([0,1])
plt.legend()
plt.grid()

#%% --- Full Simulation of countrates ---

N = 100000      # number of trials
dead = 40       # deadtime of detector
width_w = 300   # width of pulse

emitted_photons = np.random.normal(0,width_w,N)

temp_list = []
for _ in range(5):
    emitted_photons = np.random.normal(0,width_w,N)
    kickout = np.ones(len(emitted_photons), dtype = bool)
    for i,photon_1 in enumerate(emitted_photons[:-1]):
        if np.abs(photon_1 - emitted_photons[i+1]) < dead and emitted_photons[i+1]> photon_1:
            kickout[i] = False
    temp_list.append(len(emitted_photons[kickout])/N)

average = np.array(temp_list).mean()
print(f'Average detection probability: {average} %')
        
plt.figure(1)
detected_photons = emitted_photons[kickout]
count, bins, ignored = plt.hist(detected_photons, 100, density=False)
plt.plot(bins, 1/(width_w * np.sqrt(2 * np.pi)) *
         np.exp( - (bins - 0)**2 / (2 * width_w**2) )*N*(bins[1]-bins[0]),
         linewidth=2, color='r')    
   
        













