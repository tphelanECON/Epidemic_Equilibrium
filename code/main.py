"""
Produces figures used in the paper
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')
import calibration, classes, math, time

SIGMA_fine = calibration.SIGMA_fine
T_vac_list = calibration.T_vac_list

tic = time.time()
print("Benchmark case")
sigma, T_vac, aIk = 0.4, 1, 1
res = classes.results(sigma,T_vac,aIk)
classes.SIRD_plots(res)
classes.contour_plots(res)
classes.activity_plots(res)

print("Perfect testing")
sigma, T_vac, aIk = 0.99, 1, 1
res = classes.results(sigma,T_vac,aIk)
classes.SIRD_plots(res)
classes.contour_plots(res)
classes.activity_plots(res)

print("Quarantine")
sigma, T_vac, aIk = 0.4, 1, 0.4
res = classes.results(sigma,T_vac,aIk)
classes.SIRD_plots(res)
classes.contour_plots(res)
classes.activity_plots(res)

print("Late vaccine arrival")
sigma, T_vac, aIk = 0.4, 100, 1
res = classes.results(sigma,T_vac,aIk)
classes.SIRD_plots(res)
classes.contour_plots(res)
classes.activity_plots(res)
classes.DS_plots(res)

"""
Create the welfare and death toll figures
"""

print("Variation in diagnostic rate (benchmark)")
classes.robust_sigma_plots(SIGMA_fine,1,1.0)

print("Variation in diagnostic rate (quarantine)")
classes.robust_sigma_plots(SIGMA_fine,1,0.4,loc='upper right')

print("Variation in arrival rate of vaccine")
classes.robust_T_vac_plots(0.4,T_vac_list,1.0)

"""
Compare PBE and PRME allocations (this takes a while)
"""

#print("PRME_PBE")
#import PRME_PBE
toc = time.time()

print("Time for entire code:", toc-tic, "seconds")
