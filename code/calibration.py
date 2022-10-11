"""
Benchmark calibration

Discussion of the below choices is conducted in Section 4.1 of the paper.

Note that the following rates are DAILY not YEARLY. Also note that VSL isn't
necessary in what follows; only u_D is used. 
"""

import numpy as np

beta = 1/5.4        #Rai et al (2021)
gamma = 1/13.5      #You et al (2020)
r = 0.05/365.25     #5% annual dscount rate (standard)
T_vac_benchmark = 1 #expected vaccine arrival in years
delta0 = 0.0027     #IFR (not CFR), median value reported by Ioannidis (2021). CFR fraction of reported cases who die.
sigma = 0.4         #benchmark case as discussed in Section 4.1
a_bnd = [0.4,1.0]   #note lower bound is a bit arbitrary but never binds
alpha = 1           #log utility
GDP = 3.1*10**4                 #US per-capita GDP 2006.
#u_D = -10.77                   #earlier calibration (2020) for Sweden (only used in earlier version)
u_D = -12.22                    #calibrated to match Sweden's value. THIS IS THE ONE USED IN THE PAPER
VSL = (GDP/365)*(u_D)/(-r)      #implied by Sweden's value
VSL2 = 7.4*10**6                #alternative calibration from US data. From EPA (2006).
N_sigma = 30                    #number of sigma values used in Figures 4 and 5
N_T = 30                        #number of T values used in Figure 8
SIGMA_fine = list(np.linspace(0.1,0.99,N_sigma))
T_vac_list = 10**np.linspace(np.log10(0.25), 2,N_T)

"""
Grid parameters and initial conditions
"""

Imin, median, maxiter = 10**-8, 10**-4, 250
N, tol = (100,800), 10**-5

I0 = 10**-6
S0 = 1-I0
T, K, init = 600, 1, (S0, I0)
