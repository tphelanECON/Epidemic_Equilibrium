"""
Benchmark calibration
"""

import numpy as np

beta = 1/5.4        #Rai et al (2021)
gamma = 1/13.5      #You et al (2020)
r = 0.05/365.25     #5% annual dscount rate (standard)
T_vac_benchmark = 1 #expected vaccine arrival in years
delta0 = 0.0027     #IFR (not CFR), median value reported by Ioannidis. CFR fraction of reported cases who die.
sigma = 0.4         #inferred from CFR = 0.0135 reported by Johns Hopkins
a_bnd = [0.4,1.0]   #note lower bound a bit arbitrary but never binds
alpha = 1           #CRRA parameter (standard)
GDP = 3.1*10**4                 #US per-capita GDP 2006.
#u_D = -10.77                   #calibrated to match Sweden's value chosen in 2020
u_D = -12.22                    #calibrated to match Sweden's value. New value as of April 6th 2021.
VSL = (GDP/365)*(u_D)/(-r)      #calibrated to match Sweden's value
VSL2 = 7.4*10**6                #alternative calibration from US data. From EPA (2006).
N_sigma = 30
N_T = 30
SIGMA_fine = list(np.linspace(0.1,0.99,N_sigma))
T_vac_list = 10**np.linspace(np.log10(0.25), 2,N_T)

"""
Grid parameters and initial conditions
"""

Imin, median, maxiter = 10**-8, 10**-4, 200
N, tol = (100,800), 10**-5

I0 = 10**-6
S0 = 1-I0
T, K, init = 600, 1, (S0, I0)

"""
calculation of expected deaths
"""

def exp_death(nu,path):
    pv = nu*np.exp(-nu*np.arange(T*K)/K)/K
    pv_remain = 1-np.sum(pv)
    #cumulative death conditional on no vaccine:
    D_con_ult = path[1,:]*delta0 + path[3,:]
    #expectation wrt vaccine arrival:
    return np.sum(pv*D_con_ult) + pv_remain*D_con_ult[-1]
