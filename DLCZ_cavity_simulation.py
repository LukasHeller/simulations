"""
Simulation of the DLCZ cavity, allowing for enhancement of either the write, 
the read or both transitions
"""
import numpy as np
import matplotlib.pyplot as plt
import fitToolsv3
# import seaborn as sns
# sns.set() 
# sns.set_context("talk")

#%%
eta_r = 0.3
eta_w = 0.3
branching = 0.5
tau = 1
beta_w_vs_beta_r = 1
effZero = 0.4
alpha = 1
p_spinwave = np.linspace(0.005,0.1,100)
plt.close(1)
fig, (ax1,ax2,ax3) = plt.subplots(1,3, num = 1, figsize =(12,4))

params = [[[1,5,10,15],
           [1]*4, 
           ["k"]*4, 
           [eta_w*0.6]*4,
           [eta_r]*4,
           "Write enhanced, read decoupled, int.eff 40%",
           "write_res.png"],
          [[1,1/5,1/10,1/15],
           [1,5,10,15],
           ["g"]*4,
           [eta_w]*4,
           [eta_r*0.6]*4,
           "Read enhanced, write decoupled, int.eff 40%",
           "read_res.png"],
          [[1]*4,
           [1,5,10,15],
           ["r"]*4,
           [eta_w*0.6]*4,
           [eta_r*0.6]*4,
           "Both enhanced, int.eff 40%",
           "both_res.png"],
          [[1]*4,
           [1]*4, 
           ["blue"]*4, 
           [eta_w]*4,
           [eta_r]*4,
           "No enhancement int.eff 40%",
           "no.png"]]

#%%

for paraset in params:
    
    for beta_w_vs_beta_r, alpha, color, eta_w, eta_r in zip(paraset[0],paraset[1], paraset[2], paraset[3], paraset[4]):

        p_wr_coh, p_wr_incoh = fitToolsv3.p_wr(0,p_spinwave, eta_r, eta_w, branching, effZero, tau, 
                        beta_w_vs_beta_r = beta_w_vs_beta_r, alpha = alpha)
        
        p_w_coh, p_w_incoh = fitToolsv3.p_w(0, p_spinwave, eta_w)
        p_r_coh, p_r_incoh = fitToolsv3.p_r(0, p_spinwave, eta_r, branching, effZero, tau,
                        beta_w_vs_beta_r = beta_w_vs_beta_r, alpha = alpha)
        
        g2 = (p_wr_coh + p_wr_incoh)/(p_w_coh + p_w_incoh)/(p_r_coh+ p_r_incoh)
        eff = (p_wr_coh + p_wr_incoh)/(p_w_coh + p_w_incoh)
        coinc_prob = p_wr_coh + p_wr_incoh
        
        ax1.plot(p_spinwave*100, g2, ls = "-", color = color, alpha = 0.6)
        ax2.plot(p_spinwave*100, eff*100, ls = "-",color = color, alpha = 0.6)
        ax3.plot(p_spinwave*100, coinc_prob*100, ls = "-",color = color, alpha = 0.7)
    
ax3.set_ylim((0,0.6))
ax3.set_xlabel("p (%)")
ax3.set_ylabel(r"$p_{w,r}$ (%)")    
ax2.set_ylim((0,40))
ax2.set_xlabel("p (%)")
ax2.set_ylabel(r"$p_{r|w}$ (%)")
ax1.set_ylim((0,70))
ax1.set_xlabel("p (%)")
ax1.set_ylabel(r"$g^{(2)}_{w,r}$")
fig.suptitle("all")
plt.tight_layout()

# plt.savefig("all.png")


#%%

blueDark=(0,0.3,0.6)
blueLight=(0.5,0.8,1)
greenDark=(0.1,0.5,0.1)
greenLight=(0.7,1,0.5)
orangeDark=(0.9,0.7,0)
orangeLight=(1,0.8,0.6)
redDark=(0.8,0.1,0.1)
redLight=(1,0.6,0.6)

n = 1000
R = np.linspace(0.2,1,n+1)      # Intensity Reflectivity of the cavity
Losses = [1 - 0.89, 0.11+0.08]
T = 1-R

plt.close('all')
fig, ax1 = plt.subplots(num = 2, figsize = (5,3))
ax2 = ax1.twinx()

for lstyle,lwidth, L in zip(['-', '--'],[2,1.3], Losses):
    F = np.pi*(R*(1-L))**.25/(1-(R*(1-L))**.5)
    eta_escape = T/(T + L)*100
    alpha = 2*F/np.pi
    print(alpha)
    
    #---------------------------------------------
    
    p = np.zeros(len(alpha))
    
    for i,pi in enumerate(p):
        g2 = 100
        p_spinwave = 0
        
        
        # beta_w_vs_beta_r = 1
        # alpha_iter = alpha[i]
        # eta_w_iter = eta_w*0.6
        # eta_r_iter = eta_w*0.6      
           
        # beta_w_vs_beta_r = 1/alpha[i]
        # alpha_iter = alpha[i]
        # eta_w_iter = eta_w
        # eta_r_iter = eta_w*0.6
        
        # write enhanced
        beta_w_vs_beta_r = alpha[i]
        alpha_iter = 1
        eta_w_iter = eta_w*0.6
        eta_r_iter = eta_w
        
        
            
        while g2 > 20:
            
            p_spinwave+=0.0001
            
            # "Both enhanced, int.eff 40%",
            # "both_res.png"],
            

          
            p_wr_coh, p_wr_incoh = fitToolsv3.p_wr(0,p_spinwave, eta_r_iter, eta_w_iter, branching, effZero, tau, 
                                beta_w_vs_beta_r = beta_w_vs_beta_r, alpha = alpha_iter)
                
            p_w_coh, p_w_incoh = fitToolsv3.p_w(0, p_spinwave, eta_w_iter)
            p_r_coh, p_r_incoh = fitToolsv3.p_r(0, p_spinwave, eta_r_iter, branching, effZero, tau,
                            beta_w_vs_beta_r = beta_w_vs_beta_r, alpha = alpha_iter)
            
            g2 = (p_wr_coh + p_wr_incoh)/(p_w_coh + p_w_incoh)/(p_r_coh+ p_r_incoh)
        print(p_spinwave)
        p[i] = p_spinwave
    
    #----------------------------------------------
    
    alpha[:] = 1
    rate_enhance = eta_w*eta_r*(eta_escape/100)**1*p*effZero*alpha/(1+effZero*(alpha-1))
    
    ax1.plot(R*100, F, c = blueDark, ls = lstyle, lw = lwidth)
    ax1.axvline(86.2, ls = '--', color = 'grey', zorder = -1)
    ax1.plot(R*100, eta_escape, c = greenDark, ls = lstyle, lw = lwidth)
    ax1.set_xlabel('Outcoupling mirror reflectivity (%)')

    ax2.plot(R*100, rate_enhance*100, c = redDark, ls = lstyle, lw = lwidth)
    ax1.set_xlim((30,100))
    ax2.set_ylabel(r'$p_{w,r}$ (%)')

    ax1.tick_params(direction='in')
    ax1.set_position([0.12, 0.14, 0.77, 0.84])
    ax2.tick_params(direction='in')
    ax2.tick_params(axis = 'y',colors = redDark)
    ax2.yaxis.label.set_color(redDark)
    ax2.set_position([0.12, 0.14, 0.77, 0.84])
    
rate = eta_w*eta_r*0.03*effZero + eta_escape*0
ax2.plot(R*100, rate*100, c = redDark, ls = ":", lw = lwidth)

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

ybox1 = TextArea(r'Cavity Finesse $F$', textprops=dict(color=blueDark, size=11,rotation=90,ha='left',va='bottom'))
ybox2 = TextArea(r'and', textprops=dict(color='k', size=11,rotation=90,ha='left',va='bottom'))
ybox3 = TextArea(r'Escape prob. (%)', textprops=dict(color=greenDark, size=11,rotation=90,ha='left',va='bottom'))

ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)
anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.125, -0.12), bbox_transform=ax1.transAxes, borderpad=0.)

ax1.add_artist(anchored_ybox)

plt.show()
plt.savefig('output/writeenh_vs_outcoupler_20.png',dpi=300)


#%% 

finesse = np.linspace(1,25,100)
plt.close(3)
plt.figure(3)

for effZero in [0.3,0.4,0.5]:
    alphas = 2*finesse/np.pi
    plt.plot(finesse, 100*effZero*alphas/(1+effZero*(alphas-1)), label = effZero)
    
plt.legend()
plt.xlabel("Finesse")
plt.ylabel ("Efficiency (%)")
    