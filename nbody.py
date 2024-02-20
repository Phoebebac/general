# -*- coding: utf-8 -*-
"""
This code solves the n body problem using a velocity verlet integrator. It
caluculates the energy as angular momentum of the system at each timestep.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Functions
def two_bodies():
    '''Sets up the two body problem based on global initial conditions.
    Calls the plotting function which will in turn progress the system past 
    initial conditions.
    Calls functions to test if the system obeys keplers 3rd law.
    '''
    n = 2
    # Initial conditions
    mass = [small_mass,1.0]
    r_0 = 1.0  # initial distance
    p = r_0 * (1 + e)
    alpha = mass[1]  # G=1 and m1>>m0
    pos = np.array([[-p/(1 + e),0.0,0.0],[0.,0.,0.]])
    vel = np.array([[0.0,(1 + e) * (alpha / p) ** 0.5,0.0],[0.0,0.0,0.0]])
    r = np.array([[0.,1.],[1.,0.]])
    acc = acceleration(pos,r,n,mass)
    
    # preventing machine precision issues from shifting entire system
    num = 0.0
    numv = 0.0
    for j in range(0,n):  # particle of focus
        num += mass[j] * pos[j,:]  # numerator for center of mass calculation
        numv += mass[j] * vel[j,:]             
    COM = (num)/(sum(mass))
    COMV = (numv)/(sum(mass))   
    pos = pos - COM
    vel = vel - COMV
    
    x1, y1, time = plotting(pos,vel,acc,di,n,mass)
    
    # Validation
    major = axis(x1,y1)  # semi major and semi minor axis
    per = period(x1,time)  # observational period
    kepler_3(major,per,mass)  
 
    
def three_bodies():
    '''Sets up the three body problem based on global initial conditions, and 
    calls the plotting function which will in turn progress the system past 
    initial conditions.
    '''
    mass = [1.0,1.0,1.0]
    n = 3
    pos = np.array([[-1.0,0.0,0.0],[1.0,0.,0.],[0.0,0,0]])
    vel = np.array([[p1,p2,0.0],[p1,p2,0.0],[-2 * p1,-2 * p2,0.0]])
    r = distances(pos,n)
    acc = acceleration(pos,r,n,mass)
     
    # preventing machine precision issues from shifting entire system
    num = 0.0
    numv = 0.0
    for j in range(0,n): # particle of focus
        num += mass[j] * pos[j,:] # numerator for center of mass calculation
        numv += mass[j] * vel[j,:]             
    COM = (num)/(sum(mass))
    COMV = (numv)/(sum(mass))   
    pos = pos - COM
    vel = vel - COMV
         
    plotting(pos,vel,acc,di,n, mass)   


def plotting(pos, vel, acc,di,n, mass):
    '''Constructs plots for the positions of the bodies, the total angular 
    momentum in the system against time, and the percentage error in the total 
    energy of the system against time. 
    Calls the functions to progress the system from initial conditions
    
    Inputs
        pos : initial positions in xyz of the n bodies as a nx3 array
        vel : initial velocities in xyz of the n bodies as a nx3 array 
        acc : initial accelerations in xyz of the n bodies as a nx3 array
        di : number of iterations to preform
        n : number of bodies in the system
        mass : list of masses of the bodies        
    Outputs
        3 plots as specified above
        x1 : list of x positions of the 1st mass
        y1 : liist of y positions of the 1st mass
        time : list of times with data points
        
    '''
    d = 0
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    if n == 3:
        x3 = []
        y3 = []
    ang_mom = []
    energy = []
    time = [0]
    r = distances(pos,n)
    
    #total enerfy and angular momentum for initial conditions
    TES, LS = energy_and_momentum(pos,vel,r,n,mass)
    ang_mom.append(LS[2])
    energy.append(TES)
    
    # post initial values
    while d < di:
        
        d+= 1
        t=(d*dt)  # current time 
        time.append(t)
        
        pos, vel, acc,r = verlet(pos,vel,acc,n, mass)  # integrator
        TES, LS = energy_and_momentum(pos,vel,r,n,mass)  # energy and ang momentum

        x1.append(pos[0,0]) 
        y1.append(pos[0,1])
        x2.append(pos[1,0]) 
        y2.append(pos[1,1])
        if n == 3:
            x3.append(pos[2,0]) 
            y3.append(pos[2,1])
        ang_mom.append(LS[2])
        energy.append(TES)
        
    # percentage error in energy
    e_TES = error(energy)
    
    # percentage error in energy plot
    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.set_ylabel('% error in total energy')
    ax.set_xlabel('time')
    ax.plot(time,e_TES)
    plt.tight_layout()
    #plt.savefig('y1a_2.png', dpi=300)
    
    # angular momentum plot
    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.set_ylabel('total angular momentum ')
    ax.set_xlabel('time')
    ax.plot(time, ang_mom)
    plt.tight_layout()
    #plt.savefig('y1a_1.png', dpi=300)
    
    # positions of the bodies plot
    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.set_xlabel('position in x ')
    ax.set_ylabel('position in y ') 
    ax.plot(x1,y1, label='mass 1')
    ax.plot(x2,y2, label = 'mass 2')
    if n==3:
        ax.plot(x3,y3, label = 'mass 3')
    ax.legend()
    plt.tight_layout()
    #plt.savefig('y1a_0.png', dpi=300)

    return x1, y1, time


def verlet(pos, vel, acc,n, mass):
    ''' Preforms one iteration of the velocity verlet integrator
    Inputs
        pos : positions in xyz of the n bodies as a nx3 array
        vel : velocities in xyz of the n bodies as a nx3 array 
        acc : accelerations in xyz of the n bodies as a nx3 array
        n : number of bodies in the system
        mass : list of masses of the bodies
    Outputs
        the updated version of inputs pos, vel, acc
        r : the updated distance between the two bodies as a nxn array
    '''
    
    for j in range(0,n):  # particle of focus
        for i in range(0,3):  # x,y, z components
            vel[j,i] = vel[j,i] + 0.5 * dt * acc[j,i]  # velocity half step
            pos[j,i] = pos[j,i] + dt * vel[j,i]  # position full step
    
    r = distances(pos,n) # updated distance between the bodies
    acc=acceleration(pos,r,n,mass)  # acceleration full step
    
    # velocity full step
    for j in range(0,n): 
        for i in range(0,3):
            vel[j,i] = vel[j,i] + 0.5 * dt * acc[j,i]
    
    # preventing machine precision issues from shifting entire system        
    num = 0.0
    numv = 0.0
    for j in range(0,n):  # particle of focus
        num += mass[j] * pos[j,:]
        numv += mass[j] * vel[j,:]  
    # centre of mass calculation        
    COM = (num)/(sum(mass))
    COMV = (numv)/(sum(mass))
    pos = pos - COM
    vel = vel - COMV
    
    return pos,vel,acc,r
 
    
def distances(pos,n):
    '''Calculates the distance between n bodies.
    Inputs
        pos : positions in xyz of the n bodies as a nx3 array
        n : number of bodies in the system
    Outputs
        r = the updated distance between the two bodies as a nxn array.
            eg. r[0,2] and r[2,0] and the distances between bodies 0 and 2
    '''
    r=np.zeros((n,n))
    for j in range(0,n):  # particle of focus
        for k in range(0,n):  # other particles
            if k>j:
                r[j,k] = ((pos[j,0]-pos[k,0])**2 + (pos[j,1]-pos[k,1])**2 + (pos[j,2] - pos[k,2])**2)**0.5
                r[k,j]=r[j,k]  # distance between 2 & 1 = distance between 1 &2
    return r


def acceleration(pos,r,n,mass):
    '''Calculates the acceleration of each body (G=1).
    Inputs
        pos : initial positions in xyz of the n bodies as a nx3 array 
        r : distance between the two bodies as a nxn array.
        n : number of bodies in the system
        mass : list of masses of the bodies
    Outputs
        acc : accelerations in xyz of the n bodies as a nx3 array      
    '''
    acc=np.zeros((n,3))
    for j in range(0,n): # particle of focus
        for k in range(0,n): # other particles
            if k != j:
                for i in range(0,3):
                    # G=1, s=gravitational softening
                    acc[j,i] += -(mass[k] * (pos[j,i] - pos[k,i]))/(s + r[j,k]**3)
    return acc   
 

def energy_and_momentum(pos,vel,r,n,mass):
    '''Calculates the total energy and angular momentum of the system
    Inputs
        pos : positions in xyz of the n bodies as a nx3 array
        vel : velocities in xyz of the n bodies as a nx3 array 
        r : distance between the two bodies as a nxn array.
        n : number of bodies in the system
        mass : list of masses of the bodies
    Outputs
        TES = Total energy of the system
        LS = Total angular momentum of the system
    '''
    KE = np.zeros(n)
    PE = np.zeros(n)
    TE = np.zeros(n)
    L = np.zeros(n)
    TES = [0]
    num = [0]
    LS = [0]       
    
    # Energy
    for j in range(0,n): # particle of focus
        # kinetic
        KE[j] = 0.5 * mass[j] * (vel[j,0]**2 + vel[j,1]**2 + vel[j,2]**2)
        for k in range(0,n):  # other particles
            if k > j:
                PE[j] += (-mass[k] * mass[j]) / r[j,k]  # potential
                
        TE[j] = KE[j] + PE[j]  
        TES += TE[j]  # total energy of system
       
        # centre of mass calculation
        num += mass[j] * pos[j,:]   
    COM = (num)/(sum(mass))
    
    for j in range(0,n):
        L = np.cross((COM[:] - pos[j,:]), vel[j,:])
        LS += L # angular momentum of system
    return TES, LS    


def error(q):
    '''Calculates percentage error of values by comparing to the first value
    Input
        q = list of values
    Output
        error_perc = list of percentage errors
    '''
    error_perc = []
    for i in range(0,len(q)):
        error = 100 * abs((q[i] - q[0])/q[0])
        error_perc.append(error)    
    return error_perc
    

def axis(x,y):
    '''Calculates the semi major axis of the system
    Inputs
        x : x positions of the satellite body
        y : y positions of the satellite body
    Output
       major : semi major axis
    '''
    ymax = max(y)
    ymin = min(y)
    xmax = max(x)
    xmin = min(x)    

    # both semi major and semi minor axis
    r_x = (xmax - xmin)/2
    r_y = (ymax -ymin)/2
    
    major = max(r_x,r_y)
    return major


def period(lst,time):
    '''Determines the obsevational period of orbit of the satellite body by
    determining the times between when the x or y value changes from positive 
    to negative.
    
    Inputs
        lst : list of either x or y positions of the body
        time : list of times of data points
    Outputs
        period : the length of time for a full orbit
    '''
    pos_to_neg = []
    t_pos_to_neg = []
    period_lst = []
    # indices of sign change from positive to negative
    for x_val, (x_0, x_1) in enumerate(zip(lst, lst[1:])):
        if x_0 >= 0 and x_1 < 0 :
            pos_to_neg.append(x_val)
            
    # time of each indices
    for i in pos_to_neg:
        t_pos_to_neg.append(time[i])
    
    # diferences in time between successive indices    
    for i in range(0,len(pos_to_neg)-1):
        p = t_pos_to_neg[i+1] - t_pos_to_neg[i]
        period_lst.append(p)
        
    period = sum(period_lst)/len(period_lst)  # average period
    return period


def kepler_3(major,per,mass):
    '''Calculates the period using Keplers third law using the semi major axis.
    Inputs
        major : semi major axis
        per : observational period
        mass : masses of the two bodies
    Outputs
        printouts of: the obsevational period, period from keplers third law
        and the semi-major axis
    '''
    # using semimajor axis to find the period in days
    p_k = (((4 * np.pi **2 * major **3) / (sum(mass))) **0.5)
    return print('observational period =', per), print('period from Keplers 3rd Law =',p_k), print('semi major axis=',major), 
  


# Changeable initial conditions

# both 2 and 3 body systems
di = 150000  # number of iterations, for 2 body should be at least few periods
dt = 0.00005  # timestep
s = 1e-13  # gravitational softening, can be zero for 2 body
# needed for two body only
e = 0.6 # eccentricity
small_mass = 0.001  # <<1
# needed for three bodies only
p1 = 0.306893  # velocity ratio in x
p2 = 0.125507 # velocity ratio in y

# calling the functions, comment out which not using
#two_bodies()
three_bodies()

