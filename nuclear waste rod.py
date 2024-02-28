# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:40:09 2024
This was done by translating the method put forth in 'Numerical Solutions of Partial Differential Equations (2011) by Louise Olsen-Kettle,
University of Queensland' into code.
@author: Phoeb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# constants and initial conditions
r0 = 0
t0 = 0  # initial time
rc = 100  # where radius where temp is environmnet temp, cm
tf = 100  # final time in yrs
n = 100  # number of gridpoints in r direction # might need to change to 99 as n+1
m = 1000  # number of gridpoints in t direction
kappa = 2e7 # thermal diffusivity #cm2/yr
a = 25  # rod radius cm
T_env = 300  # environment temp
tau_0 = 100  # half life
T_rod = 1  # initial temperature change in rod, kelvin 
dr = rc / (n+1)  # cm 
dt = tf / m  # yrs
su = kappa * dt / dr**2 # gain parameter


def A_b_setup():
    # setting up matrix A
    j = np.arange(2,n+1)
    s = np.array([su]*n)
    A = np.diagflat([1 + 2*s]) # main diagonal
    s = np.delete(s,[-1]) # off diagonal has 1 less element
    A+=np.diagflat(-s + s/(2*j),-1) # below diagonal
    A+=np.diagflat(-s - s/(2*(j-1)),1) # above diagonal
    
    # neumann boundary  dT/dr (r=0, t) = 0
    A[0,0] = 1 + su + su/2  # first term T0 K+1 = T1 K+1
    
    b = np.zeros(n)
    # dirichlet boundary at T(r = rc , t) = T_env
    b[n-1] = -T_env * (-su - su/(2*n))  # negative here as +b later 
    
    return A,b


def source_vector(t):
    #source vector setup
    source = np.zeros(n)
    for j in range (1, n+1):
        if j*dr < a:  # within the rod
            source[j-1] = T_rod * np.exp(-t/tau_0) / a**2      
    return source
        

def solve(t, Temp_temp):
    T_tk = Temp_temp
    source = source_vector(t)
    RHS = T_tk + kappa*dt*source +b  # A*T+1
    A_inv = np.linalg.inv(A)
    T_k1 = A_inv @ RHS  # new temperature, rearranged matrix equation
    
    return T_k1
    

def loop():
    t = t0

    Temp_plot = np.zeros((n,4))  # only need to retain relevant time steps so nx4
    Temp_temp = np.zeros(n)
    
    #initial conditions T(r,t=0) = T_env
    Temp_t0 = T_env * np.ones(n)
    Temp_temp = Temp_t0
    
    f=0
    for k in range(1,m+1):
        Temp_temp = solve(t, Temp_temp)
        t = t + dt
        if k*dt==1 or k*dt==10 or k*dt==50 or k*dt==100:  # time steps to plot
            Temp_plot[:,f] = Temp_temp
            f += 1
            
    return Temp_plot, Temp_t0

A,b = A_b_setup()

Temp_plot,Temp_t0=loop()
r = np.linspace(r0+dr, rc-dr, n)

fig, ax = plt.subplots(dpi=400)
ax.set_ylabel('Temperature')
ax.set_xlabel('radius')
ax.plot(r,Temp_plot)
ax.plot(r,Temp_t0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) 
ax.legend(('1year','10years','50years','100years', '0years'))  
