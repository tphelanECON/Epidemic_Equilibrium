"""
Robustness exercise with alternative solution concept of PRME.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits import mplot3d
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator #LinearNDInterpolator very slow
import calibration, classes
import math, time

"""
Benchmark parameters:
"""

beta, gamma = calibration.beta, calibration.gamma
r, alpha = calibration.r ,calibration.alpha
delta0, sigma = calibration.delta0, calibration.sigma
a_bnd = calibration.a_bnd
GDP, u_D, VSL = calibration.GDP, calibration.u_D, calibration.VSL
Imin, median, maxiter = calibration.Imin, calibration.median, calibration.maxiter
N, tol = calibration.N, calibration.tol
exp_death = calibration.exp_death

I0, S0 = calibration.I0, calibration.S0
T, K, init = calibration.T, calibration.K, calibration.init

"""
FOLLOWING IS SPECIFIC TO THIS SCRIPT
"""

nu, aIk = calibration.nu_benchmark, 1
N, tol = (100,200), calibration.tol
N2 = (100,200,50)

"""
Create classes and solve
"""

PBE = classes.SIR_PBE(r=r,nu=nu,beta=beta,gamma=gamma,
sigma=sigma,delta0=delta0,a_bnd=a_bnd,aIk=aIk,N=N,Imin=Imin,
median=median,maxiter=maxiter,tol=tol,vsl=VSL,GDP=GDP,alpha=alpha)
ME = classes.SIR_ME(r=r,nu=nu,beta=beta,gamma=gamma,
sigma=sigma,delta0=delta0,a_bnd=a_bnd,aIk=aIk,N=N2,Imin=Imin,
median=median,maxiter=maxiter,tol=tol,vsl=VSL,GDP=GDP,alpha=alpha)

"""
Solve for equilibrium quantities and reduce ME to a 2D array
"""

tic = time.time()
a_ME = ME.solve()
U_ME = ME.solveU(a_ME)[0]
toc = time.time()
print("Time to solve PRME with N = ",ME.N)
print(toc-tic,"seconds")
tic = time.time()
a_PBE = PBE.solve()
U_PBE = PBE.Uupdate(a_PBE,a_PBE)
toc = time.time()
print("Time to solve PBE with N = ",PBE.N)
print(toc-tic,"seconds")

U_int, U_ME_2D = {}, {}
U_int = RegularGridInterpolator((ME.Igrid,ME.Sgrid,ME.mugrid),U_ME)
U_ME_2D = U_int((ME.II,ME.SS,ME.mu_eq))

I0, S0 = calibration.I0, calibration.S0
T, K, init = calibration.T, calibration.K, calibration.init

#path_PBE, a_rec_path_PBE, apath_PBE, upath_PBE = {}, {}, {}, {}
#path_ME, a_rec_path_ME, apath_ME, upath_ME = {}, {}, {}, {}

path_PBE, apath_PBE, apath_PBE, upath_PBE = PBE.simul_path(T,K,init,U_PBE,a_PBE,a_PBE)
path_ME, apath_ME, apath_ME, upath_ME = ME.simul_path(T,K,init,U_ME_2D,a_ME,a_ME)

"""
Plots
"""

fig,ax = plt.subplots()
ax.plot(np.arange(T*K)/K, apath_PBE, 'b', label='PBE', linewidth=2)
ax.plot(np.arange(T*K)/K, apath_ME, 'c', label='PRME', linewidth=2)
ax.set_title('Activity over time', fontsize=13)
ax.legend()
ax.set_xlabel('Days', fontsize=13)
destin = '../main/figures/path_PRME_PBE.pdf'
plt.savefig(destin, format='pdf', dpi=1000)
#plt.show()
#plt.close()

fig,ax = plt.subplots()
ax.plot(np.arange(T*K)/K, upath_PBE, 'b', label='PBE', linewidth=2)
ax.plot(np.arange(T*K)/K, upath_ME, 'c', label='PRME', linewidth=2)
ax.set_title('Utility over time', fontsize=13)
ax.legend()
ax.set_xlabel('Days', fontsize=13)
destin = '../main/figures/utility_PRME_PBE.pdf'
plt.savefig(destin, format='pdf', dpi=1000)
#plt.show()
#plt.close()

"""
Now compare PRME with PBE.
"""

excess = 100*(a_ME-a_PBE)/a_PBE
fig = plt.figure()
ax = plt.axes()
cp = ax.contourf(PBE.SS, PBE.II, excess*(PBE.SS + PBE.II <= 1))
fig.colorbar(cp,extend='max')
levels = [-0.0001,0.0001]
plt.contourf(PBE.SS, PBE.II, excess*(PBE.SS + PBE.II <= 1), levels=levels, colors='w')
ax.set_xlabel('Fraction of susceptible ($S$)', fontsize=13)
ax.set_ylabel('Fraction of infected ($I$)', fontsize=13)
ax.set_yscale('log')
plt.ylim(10**-3, 1)
ax.set_title('Excess activity function (%) PRME over PBE', fontsize=13)
destin = '../main/figures/contour_PRME_PBE.pdf'
plt.savefig(destin, format='pdf', dpi=1000)
plt.show()
#plt.close()
