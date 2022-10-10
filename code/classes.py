"""
This script contains class constructors and functions used in the paper.

Class constructors:

    SIR_PBE: Perfect Bayesian Equilibrium
    SIR_SPP: Social Planner's Problem
    SIR_ME: Perfect Recall Markov Equilibrium (used for appendix)

Functions:

    results(sigma,T_vac,aIk): takes sigma, T_vac and aIk as arguments and
    returns all the quantites we might want to plot. Returns "paths", "recursive",
    and "other", where
        paths = [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path]
        recursive = [C, V, a]
        other = [c_dict, time_dict, herd]
    SIRD_plots(res): takes tuple of the form res = results(sigma,T_vac,aIk) as
    agument and produces SIRD plots.
    contour_plots(res): produces contour plots.
    activity_plots(res): produces activity plots.
    DS_plots(res): produces plots for deaths and susceptible shares (conditional
    on no arrival of vaccine).
    robust_sigma_plots(sigma_list,T_vac,aIk, loc='lower right'): takes a
    list of diagnostic rates together with T_vac and aIk and produces figures for
    welfare loss and expected death toll.
    robust_T_vac_plots(sigma,T_vac_list,aIk): takes a list of vaccine arrival
    rates together with T_vac and aIk and produces figures for welfare loss and
    expected death toll.

Notation:

    r = subjective discount rate;
    T_vac = expected vaccine arrival in years;
    nu = vaccine arrival rate (inferred from T_vac, not a primitive of class);
    beta = baseline transmission rate (product of meeting rate lambda and
    transmission probability tau);
    sigma = diagnosis rate;
    gamma = recovery rate;
    delta = mortality function;
    vsl = value of a statistical life;
    GDP = Gross Domestic Product per capita for the US

Parameters coincide except N, specifying grid size, which is 3D in PRME.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time, calibration

def expGrid(a,b,c,N):
    s = (c**2-a*b)/(a+b-2*c)
    t = np.linspace(np.log(a+s),np.log(b+s),N)
    return np.maximum(0,np.exp(t)-s)

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

class SIR_PBE(object):
    def __init__(self,r=0.05/365,T_vac=1,beta=1/5.4,gamma=1/13.5,sigma=0.2,
    delta0=0.01,a_bnd=[0.01,1.0],aIk=1.0,N=(100,400),Imin=10**-8,Smin=10**-8,
    median=0.005,maxiter=20,tol=10**-3, vsl=4000000,GDP=45000, alpha = 1):
        self.beta, self.gamma, self.r = beta, gamma, r
        self.T_vac, self.nu = T_vac, (1/T_vac)/365
        self.alpha, self.sigma = alpha, sigma
        self.a_bnd, self.aIk = a_bnd, aIk
        self.delta0, self.delta = delta0, delta0/self.sigma
        self.vsl, self.GDP, self.uD = vsl, GDP, -r*vsl/(GDP/365)
        self.N, self.M, self.Delta_S = N, (N[0]+1)*(N[1]+1), 1/N[0]
        self.median, self.maxiter, self.tol = median, maxiter, tol
        self.Imin, self.Smin = Imin, Smin
        self.Sgrid = np.linspace(self.Smin,1,self.N[0]+1)
        self.Igrid = expGrid(self.Imin,1,self.median,self.N[1]+1)
        self.SS, self.II = np.meshgrid(self.Sgrid,self.Igrid)
        self.w = np.diff(self.II,axis=0)
        self.DIIp = np.vstack((self.w, self.w[-1,:]))
        self.DIIm = np.roll(self.DIIp,1,axis=0)
        self.ii,self.jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1))
        self.trans_keys = [(-1,0),(0,1),(0,-1)] #down in S, up in I, down in I
        self.V_Ik = (self.gamma*self.delta*self.uD + self.r*self.u(self.aIk))/(self.r+self.gamma) #check

    def u(self,a): #check
        if self.alpha == 1:
            return np.log(a)
        else:
            return (a**(1-self.alpha)-1)/(1-self.alpha)

    def u_inv(self,U): #check
        if self.alpha == 1:
            return np.exp(U)
        else:
            return ((1-self.alpha)*U + 1)**(1/(1-self.alpha))

    #aggregate transitions divided by Delta_t. a_tilde = average action that individual takes as given.
    def tran_func(self,ind,a_tilde):
        (ii, jj), t = ind, {}
        D = self.DIIm[jj,ii],self.DIIp[jj,ii]
        X, a_tilde = (self.SS[jj,ii],self.II[jj,ii]), a_tilde[jj,ii]
        with np.errstate(divide='ignore',invalid='ignore'):
            Sprime = -self.beta*X[0]*a_tilde*((1-self.sigma)*a_tilde + self.sigma*self.aIk)*X[1]
            t[(-1,0)] = -Sprime/self.Delta_S
            t[(0,1)] = np.maximum(-Sprime-self.gamma*X[1],0)/D[1]
            t[(0,-1)] = np.maximum(Sprime+self.gamma*X[1],0)/D[1]
        return t

    def tran_func_id(self,ind,a,a_tilde):
        ii, jj = ind
        mu = self.SS[jj,ii]/(1-self.sigma + self.sigma*self.SS[jj,ii])
        return self.sigma*mu*self.beta*a*((1-self.sigma)*a_tilde[jj,ii] + self.sigma*self.aIk)*self.II[jj,ii]

    #i is element of S, j is element of I. Corresponds to row jj*(self.N[0]+1)+ii
    def T(self,a,a_tilde):
        agg = self.tran_func((self.ii,self.jj),a_tilde)
        id = self.tran_func_id((self.ii,self.jj),a,a_tilde)
        val = -(self.r + self.nu + sum(agg.values()) + id)
        T = self.T_func(self.jj*(self.N[0]+1)+self.ii,self.jj*(self.N[0]+1)+self.ii,val)
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)))
            T = T+self.T_func(jj*(self.N[0]+1)+ii,(jj+key[1])*(self.N[0]+1)+ii+key[0],self.tran_func((ii,jj),a_tilde)[key])
        return T

    #remember reshape always reads l-to-r like a book. renamed this Vupdate.
    def Vupdate(self,a,a_tilde):
        id = self.tran_func_id((self.ii,self.jj),a,a_tilde)
        b = self.r*self.u(a) + id*self.V_Ik
        return splinalg.spsolve(-self.T(a,a_tilde), b.reshape((self.M,))).reshape((self.N[1]+1,self.N[0]+1))

    def c_func(self,V,a_tilde):
        mu = self.SS/(1-self.sigma + self.sigma*self.SS)
        return self.sigma*mu*self.beta*self.r**(-1)*(self.V_Ik - V)*self.II

    def opt_action(self,c,a_tilde):
        b = ((1-self.sigma)*a_tilde + self.sigma*self.aIk)*c
        with np.errstate(divide='ignore',invalid='ignore'):
            aFOC = (-b)**(-1/self.alpha)
            aFOC[b>=0] = self.a_bnd[1]
        return np.maximum(self.a_bnd[0],np.minimum(self.a_bnd[1],aFOC))

    def polupdate(self,V,a_tilde):
        return (self.jj<self.N[1])*self.opt_action(self.c_func(V,a_tilde),a_tilde) + (self.jj==self.N[1])*self.a_bnd[0]

    def solveV(self,a_tilde):
        V,eps,i = 1 - self.II,1,1
        a = self.a_bnd[0]*np.ones((self.N[1]+1,self.N[0]+1))
        while i < 20 and eps > 10**-6:
            a1 = self.polupdate(V,a_tilde)
            V1 = self.Vupdate(a1,a_tilde)
            eps, eps2 = np.amax(np.abs(V - V1)), np.amax(np.abs(a - a1))
            V, a, i = V1, a1, i+1
        return V, a

    #following only returns the activity function
    def solve(self):
        a_tilde = self.a_bnd[0]*np.ones((self.N[1]+1,self.N[0]+1))
        V = np.zeros((self.N[1]+1,self.N[0]+1))
        eps, i = 1, 1
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            V1, a_tilde1 = self.solveV(a_tilde)
            eps, epsU = np.max(np.abs(a_tilde - a_tilde1)), np.max(np.abs(V - V1))
            if i % 5 == 0:
                print("Outer loop iteration:", i, "supremum difference:", eps)
            a_tilde, V, i = a_tilde1, V1, i+1
        toc = time.time()
        print("Converged in", i, "iterations.", "supremum difference:", eps)
        print("Time taken:", toc-tic)
        return a_tilde

    #in following, a_rec is evaluated on path but does not affect path
    def simul_path(self,T,K,init,V,a_rec,a):
        #X = S,I,R,D; Sp defined for convenience
        X, Sp, Dt = np.zeros((4,T*K)), np.zeros((T*K,)), 1/K
        Vpath, a_rec_path, apath = np.zeros((T*K,)), np.zeros((T*K,)), np.zeros((T*K,))
        X[0,0], X[1,0] = init
        V_int = RegularGridInterpolator((self.Igrid,self.Sgrid),V,method='linear',bounds_error=False)
        a_rec_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a_rec,method='linear',bounds_error=False)
        a_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a,method='linear',bounds_error=False)
        Vpath[0] = V_int((X[1,0],X[0,0]))
        a_rec_path[0] = a_rec_int((X[1,0],X[0,0]))
        apath[0] = a_int((X[1,0],X[0,0]))
        for t in range(T*K-1):
            Sp[t] = -self.beta*X[0,t]*apath[t]*((1-self.sigma)*apath[t] + self.sigma*self.aIk)*X[1,t]
            X[0,t+1] = X[0,t] + Dt*Sp[t]
            X[1,t+1] = X[1,t] + Dt*(- Sp[t] - self.gamma*X[1,t])
            X[2,t+1] = X[2,t] + Dt*self.gamma*(1 - self.delta*self.sigma)*X[1,t]
            X[3,t+1] = X[3,t] + Dt*self.gamma*self.delta*self.sigma*X[1,t]
            with np.errstate(divide='ignore',invalid='ignore'):
                check = V_int((X[1,t+1],X[0,t+1]))+a_rec_int((X[1,t+1],X[0,t+1]))+a_int((X[1,t+1],X[0,t+1]))
                if np.isnan(check):
                    Vpath[t+1], a_rec_path[t+1], apath[t+1] = Vpath[t], a_rec_path[t], apath[t]
                else:
                    Vpath[t+1] = V_int((X[1,t+1],X[0,t+1]))
                    a_rec_path[t+1] = a_rec_int((X[1,t+1],X[0,t+1]))
                    apath[t+1] = a_int((X[1,t+1],X[0,t+1]))
        return X, a_rec_path, apath, self.u_inv(Vpath)

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

"""
Social planner's problem
"""

class SIR_SPP(object):
    def __init__(self,r=0.05/365,T_vac=1,beta=1/(5.4),gamma=1/13.5,sigma=0.15,
    delta0=0.0027,a_bnd=[0.05,1.],aIk=1.,N=(100,400),Imin=10**-8,Smin=10**-8,
    median=0.01,maxiter=50,tol=10**-3,vsl=7400000,GDP=31000, alpha = 1):
        self.beta, self.gamma, self.r = beta, gamma, r
        self.T_vac, self.nu = T_vac, (1/T_vac)/365
        self.alpha, self.sigma = alpha, sigma
        self.delta0, self.delta = delta0, delta0/self.sigma
        self.a_bnd, self.aIk = a_bnd, aIk
        self.vsl, self.GDP, self.uD = vsl, GDP, -r*vsl/(GDP/365)
        self.N, self.M, self.Delta_S = N, (N[0]+1)*(N[1]+1), 1/N[0]
        self.median, self.maxiter, self.tol = median, maxiter, tol
        self.Imin, self.Smin = Imin, Smin
        self.Sgrid = np.linspace(self.Smin,1,self.N[0]+1)
        self.Igrid = expGrid(self.Imin,1,self.median,self.N[1]+1)
        self.SS, self.II = np.meshgrid(self.Sgrid,self.Igrid)
        self.w = np.diff(self.II,axis=0)
        self.DIIp = np.vstack((self.w, self.w[-1,:]))
        self.DIIm = np.roll(self.DIIp,1,axis=0)
        self.ii,self.jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1))
        self.trans_keys = [(-1,0),(0,1),(0,-1)]

    def u(self,a):
        if self.alpha == 1:
            return np.log(a)
        else:
            return (a**(1-self.alpha)-1)/(1-self.alpha)

    def u_inv(self,U):
        if self.alpha == 1:
            return np.exp(U)
        else:
            return ((1-self.alpha)*U + 1)**(1/(1-self.alpha))

    def cost(self,aU):
        #C_bar_vac is coefficient of I in value function after the arrival of the vaccine
        C_bar_vac = self.sigma*(-self.gamma*self.delta*self.uD-self.r*self.u(self.aIk))/(self.r+self.gamma)
        return self.nu*C_bar_vac*self.II - self.gamma*self.delta*self.sigma*self.II*self.uD \
        + self.r*((self.sigma*self.SS+1-self.sigma)*(-self.u(aU)) + self.sigma*self.II*(-self.u(self.aIk)))

    def tran_func_SPP(self,ind,aU):
        (ii, jj), t = ind, {}
        D = self.DIIm[jj,ii],self.DIIp[jj,ii]
        X, aU = (self.SS[jj,ii],self.II[jj,ii]), aU[jj,ii]
        with np.errstate(divide='ignore',invalid='ignore'):
            t[(-1,0)] = self.beta*X[0]*aU*((1-self.sigma)*aU + self.sigma*self.aIk)*X[1]/self.Delta_S
            t[(0,1)] = self.beta*X[0]*aU*((1-self.sigma)*aU + self.sigma*self.aIk)*X[1]/D[1]
            t[(0,-1)] = self.gamma*X[1]/D[0]
        return t

    def T_SPP(self,aU):
        val = -(self.r + self.nu + sum(self.tran_func_SPP((self.ii,self.jj),aU).values()))
        T = self.T_func(self.jj*(self.N[0]+1)+self.ii,self.jj*(self.N[0]+1)+self.ii,val)
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)))
            T = T+self.T_func(jj*(self.N[0]+1)+ii,(jj+key[1])*(self.N[0]+1)+ii+key[0],self.tran_func_SPP((ii,jj),aU)[key])
        return T

    #the following is the solution to the first-order condition (and hence the
    #"candidate" optimum). When c<0, we take the positive root for the action.
    def cand(self,c):
        B = [1, self.sigma*c*self.aIk, 2*(1-self.sigma)*c]
        with np.errstate(divide='ignore',invalid='ignore'):
            if self.sigma==1:
                chi1 = (c >= 0)
                aU = 1/(c*self.aIk)
                aU[chi1==1] = 1
            else:
                chi1 = (c >= 0)
                aU = (-B[1] - np.sqrt(B[1]**2 - 4*B[0]*B[2]))/(2*B[2])
                aU[chi1==1] = 1
        return np.maximum(self.a_bnd[0],np.minimum(1,aU))

    def polupdate_SPP(self,C):
        #define the finite differences
        CBS = (C-np.roll(C,1))/self.Delta_S
        CFI = (np.roll(C,-1,axis=0)-C)/self.DIIp
        CBI = (C - np.roll(C,1,axis=0))/self.DIIm
        D = self.II/(self.r*(self.sigma*self.SS + 1-self.sigma))
        a = self.cand((CBS-CFI)*D*self.beta*self.SS)
        return (self.jj<self.N[1])*a + (self.jj==self.N[1])*self.a_bnd[0]

    def Cupdate_SPP(self,aU):
        return splinalg.spsolve(-self.T_SPP(aU), self.cost(aU).reshape((self.M,))).reshape((self.N[1]+1,self.N[0]+1))

    def solve(self):
        aU, C = 0*self.SS + self.a_bnd[0], self.SS + self.II
        eps, eps2, i = 1,1,1
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            aU1 = self.polupdate_SPP(C)
            C1 = self.Cupdate_SPP(aU1)
            eps, eps2 = np.amax(np.abs((aU - aU1))), np.sum(np.abs((aU - aU1)))/self.M
            if i % 5 == 0:
                print("Iteration:",i,"supremum difference:", eps)
            C, i, aU = C1, i+1, aU1
        toc = time.time()
        print("Converged in", i, "iterations.", "supremum difference:", eps)
        print("Time taken:", toc-tic)
        return self.polupdate_SPP(C)

    #this is the state-contingent value of the action below which the probability
    #of leaving the I grid vanishes.
    def a_hat(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            b0, b1 = self.beta*(1-self.sigma)*self.SS, self.beta*self.SS*self.sigma*self.aIk
            a = (-b1 + np.sqrt(b1**2 + 4*b0*self.gamma))/(2*b0)
            #a[b0==0] = #self.gamma/(self.beta*self.aIk*self.SS[b0==0])
        return np.maximum(self.a_bnd[0],np.minimum(self.a_bnd[1],a))

    def simul_path(self,T,K,init,V,a_rec,a):
        X, Sp, Dt = np.zeros((4,T*K)), np.zeros((T*K,)), 1/K
        Upath, a_rec_path, apath = np.zeros((T*K,)), np.zeros((T*K,)), np.zeros((T*K,))
        X[0,0], X[1,0] = init
        V_int = RegularGridInterpolator((self.Igrid,self.Sgrid),V,bounds_error=False)
        a_rec_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a_rec,bounds_error=False)
        a_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a,bounds_error=False)
        Upath[0] = V_int((X[1,0],X[0,0]))
        a_rec_path[0] = a_rec_int((X[1,0],X[0,0]))
        apath[0] = a_int((X[1,0],X[0,0]))
        for t in range(T*K-1):
            Sp[t] = -self.beta*X[0,t]*apath[t]*((1-self.sigma)*apath[t] + self.sigma*self.aIk)*X[1,t]
            X[0,t+1] = X[0,t] + Dt*Sp[t]
            X[1,t+1] = X[1,t] + Dt*(- Sp[t] - self.gamma*X[1,t])
            X[2,t+1] = X[2,t] + Dt*self.gamma*(1 - self.delta*self.sigma)*X[1,t]
            X[3,t+1] = X[3,t] + Dt*self.gamma*self.delta*self.sigma*X[1,t]
            with np.errstate(divide='ignore',invalid='ignore'):
                check = V_int((X[1,t+1],X[0,t+1]))+a_rec_int((X[1,t+1],X[0,t+1]))+a_int((X[1,t+1],X[0,t+1]))
                if np.isnan(check):
                    Upath[t+1], a_rec_path[t+1], apath[t+1] = Upath[t], a_rec_path[t], apath[t]
                else:
                    Upath[t+1] = V_int((X[1,t+1],X[0,t+1]))
                    a_rec_path[t+1] = a_rec_int((X[1,t+1],X[0,t+1]))
                    apath[t+1] = a_int((X[1,t+1],X[0,t+1]))
        return X, a_rec_path, apath, self.u_inv(Upath)

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

"""
Perfect recall Markov equilibrium extension conducted in the appendix.

State space now 3D: first component S, second component I, and third component
mu (beliefs). Actions of "all other agents", a_tilde, defined only on (S,I).

Equilibrium condition is then a(S,I,S/(sigma*S + 1-sigma)) = a_tilde(S,I).
"""

class SIR_ME(object):
    def __init__(self,r=0.05/365,T_vac=1,beta=1/5.4,gamma=1/13.5,sigma=0.95,
    delta0=0.01,a_bnd=[0.3,1.],aIk=1.,N=(100,200,25),Imin=10**-8,Smin=10**-8,
    median=0.01,maxiter=20,tol=5*10**-3, vsl=3700000, GDP=45000, alpha = 1):
        self.beta, self.gamma, self.r = beta, gamma, r
        self.T_vac, self.nu = T_vac, (1/T_vac)/365
        self.alpha, self.sigma = alpha, sigma
        self.a_bnd, self.aIk = a_bnd, aIk
        self.delta0, self.delta = delta0, delta0/self.sigma
        self.vsl, self.GDP, self.uD = vsl, GDP, -r*vsl/(GDP/365)
        self.N, self.M, self.Delta_S = N, (N[0]+1)*(N[1]+1), 1/N[0]
        self.median, self.maxiter, self.tol = median, maxiter, tol
        self.Imin, self.Smin = Imin, Smin
        self.Sgrid = np.linspace(self.Smin,1,self.N[0]+1)
        self.Igrid = expGrid(self.Imin,1,self.median,self.N[1]+1)
        self.SS, self.II = np.meshgrid(self.Sgrid,self.Igrid)
        self.w = np.diff(self.II,axis=0)
        self.DIIp = np.vstack((self.w, self.w[-1,:]))
        self.DIIm = np.roll(self.DIIp,1,axis=0)
        self.ii,self.jj = np.meshgrid(range(self.N[0]+1),range(self.N[1]+1))
        self.trans_keys = [(-1,0),(0,1),(0,-1)]
        self.mugrid, self.Delta_mu = np.linspace(0,1,self.N[2]+1), 1/self.N[2]
        self.mu_eq = self.SS/(1 - self.sigma + self.sigma*self.SS)
        self.V_Ik = (self.gamma*self.delta*self.uD + self.r*self.u(self.aIk))/(self.r+self.gamma)

    def u(self,a):
        if self.alpha == 1:
            return np.log(a)
        else:
            return (a**(1-self.alpha)-1)/(1-self.alpha)

    def u_inv(self,U):
        if self.alpha == 1:
            return np.exp(U)
        else:
            return ((1-self.alpha)*U + 1)**(1/(1-self.alpha))

    def tran_func(self,ind,a_tilde):
        (ii, jj), t = ind, {}
        D = self.DIIm[jj,ii],self.DIIp[jj,ii]
        X, a_tilde = (self.SS[jj,ii],self.II[jj,ii]), a_tilde[jj,ii]
        with np.errstate(divide='ignore',invalid='ignore'):
            Sprime = -self.beta*X[0]*a_tilde*((1-self.sigma)*a_tilde + self.sigma*self.aIk)*X[1]
            t[(-1,0)] = -Sprime/self.Delta_S
            t[(0,1)] = np.maximum(-Sprime-self.gamma*X[1],0)/D[1]
            t[(0,-1)] = np.maximum(Sprime+self.gamma*X[1],0)/D[1]
        return t

    #probability down in mu, probability of developing symptoms
    def tran_func_id(self,ind,a,a_tilde,k):
        ii, jj = ind
        pdmu = (1-self.sigma)*self.mugrid[k]*a[jj,ii]*self.beta*self.II[jj,ii] \
        *((1-self.sigma)*a_tilde[jj,ii] + self.sigma*self.aIk)/self.Delta_mu
        return pdmu, self.Delta_mu*(self.sigma/(1-self.sigma))*pdmu

    def T(self,a,a_tilde,k):
        agg = self.tran_func((self.ii,self.jj),a_tilde)
        id = self.tran_func_id((self.ii,self.jj),a,a_tilde,k)
        val = -(self.r + self.nu + sum(agg.values()) + id[0] + id[1])
        T = self.T_func(self.jj*(self.N[0]+1)+self.ii,self.jj*(self.N[0]+1)+self.ii,val)
        for key in self.trans_keys:
            ii, jj = np.meshgrid(range(max(-key[0],0),self.N[0]+1-max(key[0],0)), \
            range(max(-key[1],0),self.N[1]+1-max(key[1],0)))
            T = T+self.T_func(jj*(self.N[0]+1)+ii,(jj+key[1])*(self.N[0]+1)+ii+key[0],self.tran_func((ii,jj),a_tilde)[key])
        return T

    def Vupdate(self,a,Vlow,a_tilde,k):
        id = self.tran_func_id((self.ii,self.jj),a,a_tilde,k)
        b = self.r*self.u(a) + id[0]*Vlow + id[1]*self.V_Ik
        return splinalg.spsolve(-self.T(a,a_tilde,k), b.reshape((self.M,))).reshape((self.N[1]+1,self.N[0]+1))

    def c_func(self,Vlow,V,a_tilde,k):
        x = self.sigma*(self.V_Ik - V) + (1-self.sigma)*(Vlow - V)/self.Delta_mu
        return x*self.mugrid[k]*self.beta*self.r**(-1)*self.II

    def opt_action(self,c,a_tilde):
        b = ((1-self.sigma)*a_tilde+self.sigma*self.aIk)*c
        with np.errstate(divide='ignore',invalid='ignore'):
            aFOC = (-b)**(-1/self.alpha)
            aFOC[b>=0] = self.a_bnd[1]
        return np.maximum(self.a_bnd[0],np.minimum(self.a_bnd[1],aFOC))

    #U is slice of value function at mu = mu_k; Ulow is slice at mu = mu_k-1.
    def polupdate(self,Vlow,V,a_tilde,k):
        return (self.jj<self.N[1])*self.opt_action(self.c_func(Vlow,V,a_tilde,k),a_tilde) + (self.jj==self.N[1])*self.a_bnd[0]

    def solveVslice(self,Vlow,a_tilde,k):
        V,eps,i = 1 - self.II,1,1
        while i < 20 and eps > 10**-6:
            V1 = self.Vupdate(self.polupdate(Vlow,V,a_tilde,k),Vlow,a_tilde,k)
            eps = np.amax(np.abs(V - V1))
            V, i = V1, i+1
        return V

    #when mu=0 agent knows they are not susceptible and so takes highest action.
    def solveV(self,a_tilde):
        V = 0*np.ones((self.N[1]+1,self.N[0]+1,self.N[2]+1))
        a = self.a_bnd[0]*np.ones((self.N[1]+1,self.N[0]+1,self.N[2]+1))
        a[:,:,0] = (self.jj<self.N[1])*self.a_bnd[1] + (self.jj==self.N[1])*self.a_bnd[0]
        V[:,:,0] = self.u(self.a_bnd[1])
        for k in range(self.N[2]):
            V[:,:,k+1] = self.solveVslice(V[:,:,k],a_tilde,k+1)
            a[:,:,k+1] = self.polupdate(V[:,:,k],V[:,:,k+1],a_tilde,k+1)
        return V, a

    def solve(self):
        a_tilde = self.a_bnd[0]*np.ones((self.N[1]+1,self.N[0]+1))
        V = np.zeros((self.N[1]+1,self.N[0]+1,self.N[2]+1))
        eps, i = 1, 1
        tic = time.time()
        while i < self.maxiter and eps > self.tol:
            V1, a = self.solveV(a_tilde)
            a_int = RegularGridInterpolator((self.Igrid,self.Sgrid,self.mugrid),a)
            a_tilde1 = a_int((self.II,self.SS,self.mu_eq))
            d = (a_tilde - a_tilde1)*(self.SS+self.II<=1)
            eps = np.sum(np.abs(d))/self.M
            if i % 5 == 0:
                print("Outer loop iteration:",i,"L1 difference:", eps)
            a_tilde, V, i = a_tilde1, V1, i+1
        toc = time.time()
        print("Converged in", i, "iterations.", "L1 difference:", eps)
        print("Time taken:", toc-tic)
        return a_tilde

    def a_hat(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            b0, b1 = self.beta*(1-self.sigma)*self.SS, self.beta*self.SS*self.sigma*self.aIk
            a = (-b1 + np.sqrt(b1**2 + 4*b0*self.gamma))/(2*b0)
        a[b0==0] = 1
        return np.maximum(self.a_bnd[0],np.minimum(1,a))

    def simul_path(self,T,K,init,V,a_rec,a):
        X, Sp, Dt = np.zeros((4,T*K)), np.zeros((T*K,)), 1/K
        Vpath, a_rec_path, apath = np.zeros((T*K,)), np.zeros((T*K,)), np.zeros((T*K,))
        X[0,0], X[1,0] = init
        V_int = RegularGridInterpolator((self.Igrid,self.Sgrid),V,bounds_error=False)
        a_rec_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a_rec,bounds_error=False)
        a_int = RegularGridInterpolator((self.Igrid,self.Sgrid),a,bounds_error=False)
        Vpath[0] = V_int((X[1,0],X[0,0]))
        a_rec_path[0] = a_rec_int((X[1,0],X[0,0]))
        apath[0] = a_int((X[1,0],X[0,0]))
        for t in range(T*K-1):
            Sp[t] = -self.beta*X[0,t]*apath[t]*((1-self.sigma)*apath[t] + self.sigma*self.aIk)*X[1,t]
            X[0,t+1] = X[0,t] + Dt*Sp[t]
            X[1,t+1] = X[1,t] + Dt*(- Sp[t] - self.gamma*X[1,t])
            X[2,t+1] = X[2,t] + Dt*self.gamma*(1 - self.delta*self.sigma)*X[1,t]
            X[3,t+1] = X[3,t] + Dt*self.gamma*self.delta*self.sigma*X[1,t]
            with np.errstate(divide='ignore',invalid='ignore'):
                check = V_int((X[1,t+1],X[0,t+1]))+a_rec_int((X[1,t+1],X[0,t+1]))+a_int((X[1,t+1],X[0,t+1]))
                if np.isnan(check):
                    Vpath[t+1], a_rec_path[t+1], apath[t+1] = Vpath[t], a_rec_path[t], apath[t]
                else:
                    Vpath[t+1] = V_int((X[1,t+1],X[0,t+1]))
                    a_rec_path[t+1] = a_rec_int((X[1,t+1],X[0,t+1]))
                    apath[t+1] = a_int((X[1,t+1],X[0,t+1]))
        return X, a_rec_path, apath, self.u_inv(Vpath)

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape((C.size,)),(A.reshape((A.size,)),B.reshape((B.size,)))),shape=(self.M,self.M))

"""
General parameters
"""

beta, gamma = calibration.beta, calibration.gamma
r, alpha = calibration.r ,calibration.alpha
delta0 = calibration.delta0
a_bnd = calibration.a_bnd
GDP, u_D, VSL = calibration.GDP, calibration.u_D, calibration.VSL
Imin, median, maxiter = calibration.Imin, calibration.median, calibration.maxiter
N, tol = calibration.N, calibration.tol
I0, S0 = calibration.I0, calibration.S0
T, K, init = calibration.T, calibration.K, calibration.init

"""
Parameters varied in paper: sigma, T_vac, aIk. These are arguments of following.
"""

def results(sigma,T_vac,aIk):
    class_list = ['MY','PBE','SPP']
    c_dict, time_dict, herd = {}, {}, {}
    C, V, a, R0 = {}, {}, {}, {}
    SIRD, a_rec_path, a_path, V_path = {}, {}, {}, {}
    #a_SE_path requires no dictionary

    c_dict['MY'] = SIR_PBE(r=r,T_vac=T_vac,beta=beta,gamma=gamma,
    sigma=sigma,delta0=delta0,a_bnd=a_bnd,aIk=aIk,N=N,Imin=Imin,
    median=median,maxiter=maxiter,tol=tol,vsl=VSL,GDP=GDP,alpha=alpha)
    c_dict['PBE'] = SIR_PBE(r=r,T_vac=T_vac,beta=beta,gamma=gamma,
    sigma=sigma,delta0=delta0,a_bnd=a_bnd,aIk=aIk,N=N,Imin=Imin,
    median=median,maxiter=maxiter,tol=tol,vsl=VSL,GDP=GDP,alpha=alpha)
    c_dict['SPP'] = SIR_SPP(r=r,T_vac=T_vac,beta=beta,gamma=gamma,
    sigma=sigma,delta0=delta0,a_bnd=a_bnd,aIk=aIk,N=N,Imin=Imin,
    median=median,maxiter=maxiter,tol=tol,vsl=VSL,GDP=GDP,alpha=alpha)

    """
    Solve for equilibrium and efficient activity and produce:
        * cost of pandemic and utility of unknown agent in MY, PBE and SPP allocations
        * SIRD, activity and utility paths over time in MY, PBE and SPP allocations
        * recommended and static efficient actions along PBE path
    #print("Time taken for {0}:".format(c), toc-tic)
    """

    for c in ['PBE','SPP']:
        print("Now solving {0} for sigma = {1}, T = {2} and a_I = {3}:".format(c, sigma, T_vac, aIk))
        tic=time.time()
        a[c] = c_dict[c].solve()
        toc=time.time()
        time_dict[c] = toc-tic
    a['MY'] = 1 + 0*a[c]
    for c in class_list:
          C[c] = c_dict['SPP'].Cupdate_SPP(a[c])
          V[c] = c_dict['PBE'].Vupdate(a[c],a[c])
          sim = c_dict['PBE'].simul_path(T,K,init,V[c],a[c],a[c])
          SIRD[c],a_path[c],V_path[c] = sim[0], sim[2], sim[3]
          R0[c] = (c_dict[c].beta/c_dict[c].gamma)*SIRD[c][0,:]*a_path[c] \
          *(c_dict[c].sigma*c_dict[c].aIk + (1-c_dict[c].sigma)*a_path[c])
          herd_array = np.argwhere(SIRD[c][0,:]<c_dict[c].gamma/c_dict[c].beta)
          if len(herd_array)>0:
              herd[c] = herd_array[0]
          else:
              herd[c] = T
    a_rec_path = c_dict['PBE'].simul_path(T,K,init,V['PBE'],a['SPP'],a['PBE'])[1]
    S, I, PBE = SIRD['PBE'][0,:], SIRD['PBE'][1,:], c_dict['PBE']
    mu = S/(1-PBE.sigma+PBE.sigma*S)
    c = PBE.sigma*mu*PBE.beta*PBE.r**(-1)*I*(PBE.V_Ik-PBE.u(V_path['PBE']))
    a_SE_path = c_dict['SPP'].cand(c)
    """
    collect everthing together
    """
    paths = [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path]
    recursive = [C, V, a]
    other = [c_dict, time_dict, herd]
    return paths, recursive, other

"""
create SIRD plots
"""

def SIRD_plots(res):
    paths, recursive, other = res
    [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
    [C, V, a] = recursive
    [c_dict, time_dict, herd] = other
    T_vac,aIk = c_dict['PBE'].T_vac, c_dict['PBE'].aIk

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, SIRD['MY'][0,:], 'b', label='S', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['MY'][1,:], 'r', label='I', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['MY'][2,:], 'g', label='R', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['MY'][3,:], 'k', label='D', linewidth=2)
    plt.axvline(x=herd['MY'], color = 'k', linestyle = '--')
    ax.set_title('Population shares in myopic allocation', fontsize=13)
    ax.legend()
    ax.set_xlabel('Days', fontsize=13)
    ax.set_yscale('log')
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/MY_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, SIRD['PBE'][0,:], 'b', label='S', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['PBE'][1,:], 'r', label='I', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['PBE'][2,:], 'g', label='R', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['PBE'][3,:], 'k', label='D', linewidth=2)
    plt.axvline(x=herd['PBE'], color = 'k', linestyle = '--')
    ax.set_title('Population shares in equilibrium', fontsize=13)
    ax.legend()
    ax.set_xlabel('Days', fontsize=13)
    ax.set_yscale('log')
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/PBE_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, SIRD['SPP'][0,:], 'b', label='S', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['SPP'][1,:], 'r', label='I', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['SPP'][2,:], 'g', label='R', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['SPP'][3,:], 'k', label='D', linewidth=2)
    plt.axvline(x=herd['SPP'], color = 'k', linestyle = '--')
    ax.set_title('Population shares in efficient allocation', fontsize=13)
    ax.legend()
    ax.set_xlabel('Days', fontsize=13)
    ax.set_yscale('log')
    sig = round(100*round(c_dict['SPP'].sigma,1))
    destin = '../main/figures/SPP_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

"""
contour plots.
"""

def contour_plots(res):
    paths, recursive, other = res
    [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
    [C, V, a] = recursive
    [c_dict, time_dict, herd] = other
    T_vac,aIk = c_dict['PBE'].T_vac, c_dict['PBE'].aIk

    fig = plt.figure()
    ax = plt.axes()
    cp = ax.contourf(c_dict['PBE'].SS, c_dict['PBE'].II, a['PBE']*(c_dict['PBE'].SS + c_dict['PBE'].II <= 1))
    fig.colorbar(cp,extend='max')
    plt.axvline(x=c_dict['PBE'].gamma/c_dict['PBE'].beta, color = 'k', linestyle = '--', label = 'Herd immunity')
    levels = [-0.0001,0.0001]
    plt.contourf(c_dict['PBE'].SS, c_dict['PBE'].II, a['PBE']*(c_dict['PBE'].SS + c_dict['PBE'].II <= 1), levels=levels, colors='w')
    ax.set_xlabel('Fraction of susceptible ($S$)', fontsize=13)
    ax.set_ylabel('Fraction of infected ($I$)', fontsize=13)
    ax.set_yscale('log')
    plt.ylim(10**-3, 1)
    ax.set_title('Equilibrium activity function', fontsize=13)
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/PBE_contour_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig = plt.figure()
    ax = plt.axes()
    cp = ax.contourf(c_dict['SPP'].SS, c_dict['SPP'].II, a['SPP']*(c_dict['SPP'].SS + c_dict['SPP'].II <= 1))
    fig.colorbar(cp,extend='max')
    plt.axvline(x=c_dict['PBE'].gamma/c_dict['PBE'].beta, color = 'k', linestyle = '--', label = 'Herd immunity')
    levels = [-0.0001,0.0001]
    plt.contourf(c_dict['SPP'].SS, c_dict['SPP'].II, a['SPP']*(c_dict['SPP'].SS + c_dict['SPP'].II <= 1), levels=levels, colors='w')
    ax.set_xlabel('Fraction of susceptible ($S$)', fontsize=13)
    ax.set_ylabel('Fraction of infected ($I$)', fontsize=13)
    ax.set_yscale('log')
    plt.ylim(10**-3, 1)
    ax.set_title('Efficient activity function', fontsize=13)
    sig = round(100*round(c_dict['SPP'].sigma,1))
    destin = '../main/figures/SPP_contour_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

"""
activity plots, both over time and along equilibrium paths
"""

def activity_plots(res):
    paths, recursive, other = res
    [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
    [C, V, a] = recursive
    [c_dict, time_dict, herd] = other
    T_vac,aIk = c_dict['PBE'].T_vac, c_dict['PBE'].aIk

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, a_path['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(np.arange(T*K)/K, a_path['SPP'], 'k', label='Efficient', linewidth=2)
    ax.set_title('Activity levels over time', fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlabel('Days', fontsize=13)
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/a_time_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, a_path['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(np.arange(T*K)/K, a_rec_path, 'k', label='Efficient', linewidth=2)
    ax.plot(np.arange(T*K)/K, a_SE_path, 'k--', label='Static efficient', linewidth=2)
    ax.set_title('Activity levels on equilibrium path', fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlabel('Days', fontsize=13)
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/a_eq_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

"""
paths for cumulative dead and susceptible shares conditional on no arrival of vaccine
"""

def DS_plots(res):
    paths, recursive, other = res
    [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
    [C, V, a] = recursive
    [c_dict, time_dict, herd] = other
    T_vac,aIk = c_dict['PBE'].T_vac, c_dict['PBE'].aIk

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, 10**5*SIRD['MY'][3,:], 'c', label='Myopic', linewidth=2)
    ax.plot(np.arange(T*K)/K, 10**5*SIRD['PBE'][3,:], 'b', label='Equilibrium', linewidth=2)
    ax.plot(np.arange(T*K)/K, 10**5*SIRD['SPP'][3,:], 'k', label='Efficient', linewidth=2)
    ax.set_title('Cumulative deaths per 100,000', fontsize=13)
    ax.legend()
    ax.set_xlabel('Days', fontsize=13)
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/D_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(np.arange(T*K)/K, SIRD['MY'][0,:], 'c', label='Myopic', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['PBE'][0,:], 'b', label='Equilibrium', linewidth=2)
    ax.plot(np.arange(T*K)/K, SIRD['SPP'][0,:], 'k', label='Efficient', linewidth=2)
    ax.plot(np.arange(T*K)/K, gamma/beta + 0*SIRD['PBE'][0,:], 'k--', label='Herd immunity', linewidth=2)
    ax.set_title('Susceptible shares', fontsize=13)
    ax.legend()
    ax.set_xlabel('Days', fontsize=13)
    sig = round(100*round(c_dict['PBE'].sigma,1))
    destin = '../main/figures/S_sigma{0}T_vac{1}aIk{2}.pdf'.format(sig,T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

"""
welfare and expected death toll as diagnostic rate varies
"""

def robust_sigma_plots(sigma_list,T_vac,aIk, loc='lower right'):
    class_list = ['MY','PBE','SPP']
    data_W, data_con_D, data_ED = {}, {}, {}
    for c in class_list:
        data_W[c], data_con_D[c], data_ED[c] = [], [], []

    for i in range(len(sigma_list)):
        print("sigma = {0}".format(sigma_list[i]))
        paths, recursive, other = results(sigma_list[i],T_vac,aIk)
        [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
        [C, V, a] = recursive
        [c_dict, time_dict, herd] = other
        #no need to round the following
        for c in class_list:
            data_W[c].append(100*(1-V_path[c][0]))
            data_con_D[c].append(10**5*SIRD[c][3,T*K-1])
            data_ED[c].append(10**5*exp_death(c_dict['PBE'].nu,SIRD[c]))

    fig,ax = plt.subplots()
    ax.plot(sigma_list, data_W['MY'], 'c', label='Myopic', linewidth=2)
    ax.plot(sigma_list, data_W['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(sigma_list, data_W['SPP'], 'k', label='Efficient', linewidth=2)
    ax.legend(loc=loc)
    ax.set_ylabel('%', fontsize=13)
    ax.set_xlabel('$\sigma$', fontsize=13)
    ax.set_title('Welfare loss', fontsize=13)
    destin = '../main/figures/welfare_robust_sigma_T{0}aIk{1}.pdf'.format(T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(sigma_list, data_ED['MY'], 'c', label='Myopic', linewidth=2)
    ax.plot(sigma_list, data_ED['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(sigma_list, data_ED['SPP'], 'k', label='Efficient', linewidth=2)
    ax.legend(loc=loc)
    ax.set_xlabel('$\sigma$', fontsize=13)
    ax.set_title('Expected death toll (per 100,000)', fontsize=13)
    destin = '../main/figures/exp_death_robust_sigma_T{0}aIk{1}.pdf'.format(T_vac,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

"""
welfare and expected death toll as vaccine arrival varies
"""

def robust_T_vac_plots(sigma,T_vac_list,aIk):
    class_list = ['MY','PBE','SPP']
    data_W, data_con_D, data_ED = {}, {}, {}
    for c in class_list:
        data_W[c], data_con_D[c], data_ED[c] = [], [], []

    for t in range(len(T_vac_list)):
        print("Time until vaccine arrives = {0}".format(T_vac_list[t]))
        paths, recursive, other = results(sigma,T_vac_list[t],aIk)
        [SIRD, a_rec_path, a_path, V_path, R0, a_SE_path] = paths
        [C, V, a] = recursive
        [c_dict, time_dict, herd] = other
        #no need to round the following
        for c in class_list:
            data_W[c].append(100*(1-V_path[c][0]))
            data_con_D[c].append(10**5*SIRD[c][3,T*K-1])
            data_ED[c].append(10**5*exp_death(c_dict['PBE'].nu,SIRD[c]))

    fig,ax = plt.subplots()
    ax.plot(T_vac_list, data_W['MY'], 'c', label='Myopic', linewidth=2)
    ax.plot(T_vac_list, data_W['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(T_vac_list, data_W['SPP'], 'k', label='Efficient', linewidth=2)
    ax.legend(loc='lower right')
    ax.set_ylabel('%', fontsize=13)
    ax.set_xlabel('$T$', fontsize=13)
    ax.set_xscale('log')
    ax.set_title('Welfare loss', fontsize=13)
    destin = '../main/figures/welfare_robust_T_sigma{0}aIk{1}.pdf'.format(sigma,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')

    fig,ax = plt.subplots()
    ax.plot(T_vac_list, data_ED['MY'], 'c', label='Myopic', linewidth=2)
    ax.plot(T_vac_list, data_ED['PBE'], 'b', label='Equilibrium', linewidth=2)
    ax.plot(T_vac_list, data_ED['SPP'], 'k', label='Efficient', linewidth=2)
    ax.legend(loc='lower right')
    ax.set_xlabel('$T$', fontsize=13)
    ax.set_xscale('log')
    ax.set_title('Expected death toll (per 100,000)', fontsize=13)
    destin = '../main/figures/exp_death_robust_T_sigma{0}aIk{1}.pdf'.format(sigma,aIk)
    plt.savefig(destin, format='pdf', dpi=1000)
    #plt.show()
    plt.close('all')
