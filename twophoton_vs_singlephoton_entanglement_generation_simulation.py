"""
Simulation of the entanglement probability, comparing...

1) Assynchronous generation in two seperate arms, single-photon interference
2) Generation in a single arm, two-photon interference
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as con
import pdb

#%% Simulation
r = 100000
p = np.linspace(1,99,99)
success =[]

for prop in p:
    l = [0]*(100-int(prop)) + [1]*int(prop)
    flag1, flag2, flag3 = False, False, False
    s1 = 0
    s1_l = []
    for i in range(r):
        s1 += 1
        if random.choice(l):
            flag1 = True
        if random.choice(l):
            flag2 = True
        # if random.choice(l):
        #     flag3 = True
        # if flag1 and flag2 and flag3:
        if flag1 and flag2:
            s1_l.append(s1)
            flag1, flag2, flag3 = False, False, False
            s1 = 0
        # s2 += random.choice(l)
    success.append(1/(sum(s1_l)/len(s1_l))*100)
# print(f"Possibility: {1/(sum(s1_l)/len(s1_l))*100} %")
#%% Plotting
    
plt.plot(p/1, success, label = "dual-rail")
plt.plot(p/1, p*p/100, label = "two-photon")
plt.xlabel("Prob of single-photon click (%)")
plt.ylabel("Prob of Entanglement (%)")
plt.legend()
plt.grid()