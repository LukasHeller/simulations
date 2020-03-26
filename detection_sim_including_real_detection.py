# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

delta_p = 0.01
p =np.arange(delta_p,1,delta_p*5)
excitations = 24

def excite(n=10):
    powers = range(1,n)
    for power in powers:
        yield lambda x: x**power*(1-x), power
        
def excited_photons(n=10):
    powers = range(1,n)
    for power in powers:
        yield lambda x: x**power*(1-x)*power, power
    
for i, power in excite(5):
    plt.plot(p,i(p)*100, label = str(power))

plt.figure(0)
plt.legend(title = r'# of excitations ...')
plt.xlabel('Excitation probability p')
plt.ylabel('Contribution of n\'th component [%] ')

#%% Simulation - correct?

total_trials = 10000
eta_trans = 0.8
pulse_width = 450
dead = 40
measured_p_w_array = []

for p_test in p:
    photons = []
#     Photon generation
    photons_histo = []
    for exc, power in excite(excitations):
        trials_with_n_photons_per_mode = int(total_trials*exc(p_test))
        if trials_with_n_photons_per_mode:
            photons.append(np.ones([trials_with_n_photons_per_mode, power], dtype = bool))
    
    total_photons = sum([i.sum() for i in photons])
    for mode in photons:
        photons_histo += list(mode.sum(axis = 1))
    
    # Transmittion
    passed_histo = []
    for i, mode in enumerate(photons):
        dim = np.shape(mode)
        loss = np.array([[np.random.choice([True,False],p=[eta_trans, (1-eta_trans)])
                        for i in range(dim[1])] for j in range(dim[0])])
        photons[i] = mode & loss
        passed_histo += list(photons[i].sum(axis = 1))
    
    # Detection
    detected_histo = []
    for i, mode in enumerate(photons):
        dim = np.shape(mode)
        trials, power = np.shape(mode)
        pulse = np.random.normal(0,pulse_width,dim)
        for trial in range(trials):
            for photon in range(power):
                if mode[trial, photon]:
                    mask = [True]*power
                    mask[photon] = False
                    delete = (pulse[trial,:]-dead<pulse[trial, photon]) & (pulse[trial,:]>pulse[trial,photon])
                    photons[i][trial,:] = ~delete & photons[i][trial,:]
        detected_histo += list(photons[i].sum(axis = 1))
    
    measured_p_w = sum([i.sum() for i in photons])/total_trials
    measured_p_w_array.append(measured_p_w)
    print(f' --- measured_p_w: {measured_p_w:.03f} ---')

plt.plot(p, measured_p_w_array)
#%% Plotting
    
plt.figure(1)
plt.hist(photons_histo, bins = range(excitations+2), 
         label = 'After cloud',histtype=u'step')
plt.hist(passed_histo, bins = range(excitations+2),
         label = 'Before Detection',histtype=u'step')
plt.hist(detected_histo, bins = range(excitations+2), 
         label = 'After Detection',histtype=u'step')
plt.legend()
plt.show()


#    photons[i] = component & loss
#    passed_histo += list(photons[i].sum(axis = 1))









