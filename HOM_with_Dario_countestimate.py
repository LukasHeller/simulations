"""
HOM between Dario SPDC idler and my write photon
This does only require beam splitter counts, and no time bin photons
However, 4fold heralding is required
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = {"bsplit": 0.5,         # Beam splitter ratio
     
     "eta_606": 0.5,        # 606 nm detector quantum efficiency
     "eta_1436": 0.8,       # 1436 nm detector quantum efficiency
       # "eta_1436": 0.8*0.5,       # 1436 nm detector quantum efficiency
     "eta_780": 0.5,        # 780 nm detector quantum efficiency
     
     "eta_D_int": 0.45,     # DLCZ intrinsic retrieval efficiency
     "eta_D_cou_r": 0.85,   # DLCZ read coupling efficiency
     "eta_D_tr_r": 0.95,    # DLCZ read trans. eff. incl. filters
     "eta_D_cou_w": 0.75,   # DLCZ write coupling efficiency
     "eta_D_tr_w": 0.95*0.3,# DLCZ write trans. eff. incl. filters
     "eta_D_cav": 0.8,     # DLCZ cavity escape efficiency
     "alpha": 2*10/np.pi,           # DLCZ read enhancement
     "D_D": 1.5/(9+1.5),    # DLCZ duty cicle
     "R_D": 1/(1.5*10**-6), # DLCZ write trial repetition rate
     "p_spin": 0.01,        # DLCZ Spinwave creation prop.
     
      "eta_QFC": 0.15,       # QFC total efficiency fiber-to-fiber
     
     "eta_SPDC_cou_s": 0.7, # SPDC signal coupling efficiency
     "eta_SPDC_tr_s": 1,    # SPDC signal trans. eff. incl. filters
     "eta_SPDC_cou_i": 0.7, # SPDC idler coupling efficiency
     "eta_SPDC_tr_i": 0.5,  # SPDC idler trans. eff. incl. filters
     "eta_SPDC_cav": 0.5,   # SPDC cavity escape efficiency
     "g_SPDC": 2/10,      # SPDC idler autocorrelation
     "D_SPDC": 0.5,         # SPDC duty cicly
     
     # Probability of having SPDC photon pair generated per second
     # There is a factor 2 at the end which should be 1/D_SPDC
     "p_pair": 800/(0.8*0.5*0.5*0.7*0.5*0.5*0.7)*2,     
     
     "eta": 0.9            # Photon overlap factor  
}           


#%% Simulation function definition

def sim(p, **kwarg):

    for k in kwarg:
        p[k] = kwarg[k]
    
    # Intrinsic DLCZ retrieval efficiency with read cavity
    eta_D_int_eff = p["eta_D_int"]*p["alpha"]/(1+p["eta_D_int"]*(p["alpha"]-1))
    
    # Multiplicative detection efficiency
    eta_det = np.sqrt(p["bsplit"]*(1-p["bsplit"])*p["eta_1436"]**2)
    
    # Probability to find a heralded DLCZ write photon in front of the beam splitter 
    p_D = p["p_spin"]*p["eta_D_cou_w"]*p["eta_D_tr_w"]*p["eta_D_cav"]**2*\
          eta_D_int_eff*p["eta_D_tr_r"]*p["eta_D_cou_r"]*p["eta_QFC"]*p["eta_780"]
    
    # Probability to find an heralded idler photon in front of the beam splitter
    p_SPDC = p["p_pair"]*p["eta_SPDC_cav"]**2*p["eta_SPDC_cou_i"] * \
             p["eta_SPDC_tr_i"]*p["eta_SPDC_tr_s"]*p["eta_SPDC_cou_s"]*p["eta_606"]*400e-9    
      
    # DLCZ cross correlation function g_wr
    g_D_wr = 1 +(eta_D_int_eff*(1-p["p_spin"]))/(p["p_spin"]*eta_D_int_eff + \
            + p["p_spin"]*(1-eta_D_int_eff)*0.5)
    
    # DLCZ autocorrelation function g_rr,w, based on the crosscorrelation
    g_D = 4/(g_D_wr-1)
    
    # SPDC autocorrelation function g_rr
    g_SPDC = p["g_SPDC"]
    
    # Probability of coincidence click per trial, indistinguishable case
    p_12 = p_SPDC**2*eta_det**2*g_SPDC + p_D**2*eta_det**2*g_D + 2*eta_det**2*p_SPDC*p_D*(1-p["eta"])
    
    # Probability of coincidence click per trial, distinguishable case
    p_12_0 = p_SPDC**2*eta_det**2*g_SPDC + p_D**2*eta_det**2*g_D + 2*eta_det**2*p_SPDC*p_D
    
    # Visibility
    V = 1-p_12/p_12_0
    
    # Mysterious parameter of coincidence count rate in distinghuishable case.
    X = p["D_D"]*p["D_SPDC"]*p["R_D"]*p_12_0
    
    return g_D, g_D_wr,V,X,p_12_0,p_12, p_D, p_SPDC

#%% Some simulations

p_spinwave = np.linspace(0.005,0.2,100)      # Detected DLCZ write photon rate per second
plt.close(1)
fig, ax = plt.subplots(1,3, num = 1, figsize = (12,4))

para = "eta_QFC" #[0.99,0.9,0.8]

    
for scanner in [0.15]:
    
    p_c = p.copy()    
    
    p_c[para] = scanner
    label = str(scanner)

    d = pd.DataFrame(np.array([sim(p_c, **{"p_spin":p}) for p in p_spinwave]), 
                     columns = ["g_D", "g_D_wr","V","X","p_12_0","p_12", "p_D", "p_SPDC"])
    print(d.max())
    ax[0].plot(d["X"]*3600,d["V"], label = label)
    ax[0].set_xlabel("Coincidences dist. case (1/h)")
    ax[0].set_ylabel("Visibility")
    
    ax[2].plot(d["X"]*3600,d["g_D"], label = label)
    ax[2].set_xlabel("Coincidences dist. case (1/h)")
    ax[2].set_ylabel(r"$g_{r,r}^{(2)}$ DLCZ")
    
    ax[1].plot(p_spinwave*100,d["V"], label = label)
    ax[1].set_xlabel("p DLCZ (%)")
    ax[1].set_ylabel("Visibility")

ax[0].set_xlim((0,75))
ax[2].set_xlim((0,75))
ax[0].legend(title = para)
plt.tight_layout()
plt.show()

plt.savefig("output/sim.png", dpi = 300, bbox_inches = "tight")







