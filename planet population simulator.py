
"""
Created on Wed Nov  8 12:36:37 2023

@author: Phoeb
"""


# imports
#%%
# snowline on - dust surface density doubled outside of snowline in pebble predictor to double flux
from PPsnowline_drift import pebble_predictor
import numpy as np
from astropy import constants as c
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
import OL18
import winsound
import sys
#%%

# some useful constants in cgs
#%%
year = 365.25*24*3600
au = c.au.cgs.value
MS = c.M_sun.cgs.value
ME = c.M_earth.cgs.value
k_b = c.k_B.cgs.value
m_p = c.m_p.cgs.value
Grav = c.G.cgs.value
AH2 = 2.e-15
#%%

# Functions

# interpolators
def find_flux(r_interp, t_interp,flux):
    """interpolates flux between gridpoints
    inputs [in cgs]
        radial location
        time
        flux carried through from the pebble predictor into the massgrowth 
    outputs 
        interpolated flux [earthmasses per year]
    """
    # pebble predictors grid and flux values
    x=rgrid/au  # to au
    y=timegrid/year  # to year
    z=flux/ME*year  # to earthmass/year
    
    # converted imputs of needed position and time to years and au, grid and flux already sorted converted in xyz
    interp= float(RegularGridInterpolator((y,x),z,  bounds_error=False, fill_value=None)([t_interp/year, r_interp/au ] ))
    return interp

def find_st(r_interp, t_interp,st):
    """interpolates stokes number between gridpoints
    inputs [in cgs]
        radial location
        time
        st carried through from the pebble predictor into the massgrowth 
    outputs 
        interpolated stokes number [dimensionless]
    """
    
    # pebble predictors grid and flux values
    x=rgrid/au  # au
    y=timegrid/year  # years
    z=st  # dimentionless
    
    # converts position and time of planet into human units, gridpoints x,y are already converted and z is dimensionless
    interp= float(RegularGridInterpolator((y,x),z,bounds_error=False, fill_value=None)([t_interp/year, r_interp/au]))
    return interp

def find_eta(r_interp,eta_pp):
    """interpolates the pressure gradient between gridpoints
    inputs [in cgs]
        radial location
        eta carried through from the pebble predictor in the plotting function
    outputs 
        interpolated pressure gradient [dimensionless]
    """
    # pebble predictor gridpoints and pressure gradient
    x=rgrid/au
    z=eta_pp
    
    # input radial location is converted to au
    interp=float(interpolate.interp1d(x,z, bounds_error=False, fill_value="extrapolate")( r_interp/au))
    return interp

def find_sigma_g(r_interp, SigmaGas):
    "imputs and outputs cgs"
    
    x=rgrid
    z=SigmaGas
    
    interp=float(interpolate.interp1d(x,z, bounds_error=False, fill_value="extrapolate")(r_interp))
    return interp

# other functions

def density_calc(m_rock,m_ice):
    """Caluclates the solid density of the planet in units of uncompressed earthmass
    """
    vol_pl= ((((2/3) *m_rock * ME)/rhor) + ((1/3 ) *m_rock * ME)/rhom)+ ((m_ice * ME)/rhoi) # volume of planet cm^3  
    rhopl= (((m_rock+m_ice) * ME) / vol_pl) / rhoearth
    return rhopl

def gas_regime(M,isomass,loc,Sigma_g,u_gas,hgas):
    """Determines the regime for gas gas accretion
    Inputs
        M: mass of the planet [earthmass]
        isomass : isolation mass [earthmass]
        loc: location of the planet [au]
        Sigma_g: interpolated gas density at the location of the planet [gcm^-2]
        u_gas : velocity of the gas at the location of the planet [cms^-1]
        hgas: gas scale height [dimensionless]
    Outputs
        Mdot_disk: gas accretion rate [earthmass per year]    
    """
    
    Mdot_gas= (-2 * np.pi * loc*au * u_gas * Sigma_g ) /ME*year #cgs to human
    Mdot_kh = 1e-5 * (M/10)**4 * (0.005/0.1) #human
    #human
    Mdot_hill = 1.5 * 1e-3 * (hgas/0.05)**4 * (M/10)**(4/3) * (alpha/0.01)**-1 * (Mdot_gas/(MS/ME)/1e-8) * (1/(1 + (M/2.3*isomass)**2))

    Mdot_disk = min(Mdot_gas, Mdot_kh, Mdot_hill)
    return Mdot_disk

def migration(loc,dt,SigmaGas,hgas,M,isomass):  
    """Calcultates the migration of the planet. 
    Inputs
        loc: current location of the planet [human]
        dt: timestep [human]
        SigmaGas: gas surface density [cgs] 
        hgas: gas scale height [dimensionless]
        M: planetary mass [human]
        isomass: isolation mass of the planet at currentlocation [human]
    Outputs
        loc: new location of the planet [human]
        abs(v_mig): absolute value of the migration velocity [human]
    """

    if loc>rin_mig:

        v_k=np.sqrt(Grav * m_host / (loc*au)) #cgs
        Sigma_g=find_sigma_g(loc*au, SigmaGas) #cgs imput and output
        
        # type one migration
        v_1 = (-k *  hgas**(-2) * (M/Mstar) * (Sigma_g * (loc*au)**2 /m_host) * v_k) /au*year
        #human  nu     nu           nu           cgs                       cgs    human conversion, # M,Mstar=human, m_host=cgs

        # type two migration
        vel_mig= v_1 / (1+ (M/(2.3*isomass))**2) 
        
        dloc=vel_mig * dt
        loc += dloc
        
        if loc < rin_mig: # correcting for if gone beyond migration boundary
            loc = rin_mig
        
    else:
        vel_mig=0
    
    return loc, abs(vel_mig)


def massgrowth_pop(initial_mass,initial_time,initial_loc,st,flux,eta_pp):  
    """gives the growth of a protoplanet injected ,at a specific time and location, into a protoplanetary disk .
    Inputs
        initialmass: starting mass on protoplanet [earth mass]
        initialtime: injection time of protoplanet [year]
        initial_oc: injection radius of protoplanet [au]
        st: stokes number from pebblle predictor [dimensionless]
        flux: flux from pebble predictor [g/s]
        eta_pp: pressure gradient from pebble predictor [dimensionless]
    Outputs
        M: final mass of planet [earth mass]
        loc: final location of planet [au]
        rhopl: final solid density of planet [uncompressed earth density]
    """

    t=initial_time
    M=initial_mass
    loc=initial_loc
    
    dt=0
    v_mig=1e-8
    Mdot_disk = 0.1 # needed for first timestep only
    Mg=0
    
    #initial mass composition
    if loc >= r_snow:
        m_ice=0.5*M
        m_rock=m_ice
    else:
        m_rock=M
        m_ice=0
        

    while t < endtime/year: # converted to year
        
        # calculating isolation mass x= fraction of solar mass
        T = x**0.5* 280 * (loc)**(-0.5)  # temperature profile 
        hgas=np.sqrt(k_b*T*loc*au/(2.3*Grav*m_host*m_p))  # hgas is dimensionless, imput variables all cgs
        isomass = 40*x*(hgas/0.05)**3  # in eath masses
        
        #timesteps
        if v_mig>0:
            dtt=np.minimum(t/10, loc/(10*v_mig))
        else:
            dtt = t/10

      
        if M<isomass: 
            
            #interpolators
            eta=find_eta(loc*au,eta_pp)  # radial position in cgs and eta is dimentionless
            fflux=find_flux(loc*au,t*year,flux)  #imputs cgs into findflux but it outputs in human         
            tau=find_st(loc*au,t*year,st)  # imputing cgs, outputs au,year; stokes numberis dimentionless
            # efficiency
            qp=M/Mstar # planet-star mass ratio
            #planet mass/ location, human units induvidually dimensionless over all
            Rp= (((3 /(4*np.pi)))*(((2/3) * m_rock /rho_rock) + ((1/3) *m_rock /rho_metal)+ (m_ice/rho_ice)))**(1/3)/loc
            # http://adsabs.harvard.edu/abs/2018A%26A...615A.138L for paper on epsilom_general
            eps=OL18.epsilon_general( qp=qp,tau=tau, alphaz=alphaz, hgas=hgas, eta=eta, Rp=Rp)  
            
            dt=np.minimum((mass_step*M)/(eps*fflux), dtt)  # all values in human units
            
            # calculating the new mass
            dM = eps*(fflux)*(dt)
            M +=  dM 
            t += dt
            
            if M>isomass:  # correcting if goes above isolation mass
                m_extra = M - isomass
                M = isomass
                dM -= m_extra
                
            # composition
            if loc >= r_snow: # outside snowline 1/2 mass is rock and 1/2 ice
                m_ice += 0.5*dM
                m_rock = m_ice
            else:  # inside snowline all mass is rock
                m_rock += dM
                
            #changing location
            if v_mig>0:
                loc, v_mig=migration(loc,dt,SigmaGas,hgas,M,isomass)
           
            
        else: # post isolation mass
        
            dt=np.minimum((mass_step*M)/(Mdot_disk), dtt) #years
            Sigma_g=find_sigma_g(loc*au, SigmaGas) #cgs
            c_s = np.sqrt(k_b*T/(2.3 * m_p))  #cgs
            u_gas = -3/2 * alpha * hgas * c_s  #cgs
            
            Mdot_disk = gas_regime(M,isomass,loc,Sigma_g,u_gas,hgas) #human
            
            #calculating new mass
            dM_gas = Mdot_disk * dt
            Mg += dM_gas
            M +=  dM_gas 
            t += dt
            
            if v_mig>0:
                loc, v_mig=migration(loc,dt,SigmaGas,hgas,M,isomass)
 
    rhopl=density_calc(m_rock, m_ice)
    print(M-Mg)
    
    return M, loc, rhopl #, loc, m_ice/M,
  

def planet_pop(initial_mass):
    mass=[]
    densities=[]
    location=[]
    fig, ax = plt.subplots(dpi=400)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Mass [M$_{\oplus}$]')
    ax.set_xlabel('Orbital Radius [Au]')

    # https://github.com/astrojoanna/pebble-predictor/tree/v1.0 for pebble predictor code
    st_pp,flux_pp,eta_pp=pebble_predictor(rgrid=rgrid,tgrid=timegrid,
                                                 Mstar=m_host,SigmaGas=SigmaGas,
                                                 T=T,SigmaDust=SigmaDust,
                                                 alpha=alpha,vfrag=vfrag,
                                                 rhop=rhop)
    
    for r in np.logspace(-1, 1.5, num=5, endpoint=True, base=10.0):
        print('new_loc')
        for t in np.logspace(0, 6, num=5, endpoint=True, base=10.0):
            mass_final,loc_final, density_av = massgrowth_pop(initial_mass, t, r, st_pp,flux_pp,eta_pp)
                
            mass.append(mass_final)
            location.append(loc_final)
            densities.append(density_av)

    
    
    f=np.linspace(rin_mig,10**1.5,20)
    isomass = 40*x*((np.sqrt(k_b*(x**0.5* 280)*(f**0.5)*au/(2.3*Grav*m_host*m_p)))/0.05)**3
    ax.plot(f,isomass, linestyle='dashed',linewidth=0.7, c='orange', label='Pebble Isolation Mass')
    
    ax.axvline(r_snow, linestyle='dashed',linewidth=0.7, c='brown', label='Snowline')
    
    im=ax.scatter(location, mass,s=3, c=densities,
               cmap='brg')
    
    ax.scatter(0.02262, 13.15, marker="x", c='k', label='LHS 3154b', s=15)
    #fig.colorbar(im, ax=ax, label='Density [uncompressed rho$\oplus$]')
    fig.colorbar(im, ax=ax, label=r'Solid Density [Uncompressed $\rho _{\oplus}$]')

    ax.legend(fontsize='small')
    
    
#Variables
#%% 

mass_step=0.01  # percentage of growth in mass to determine timestep

# migration variables
Beta=1
Xi = 0.5
k = 2*(1.36 + 0.62*Beta + 0.43*Xi)

# star properties
#x=0.1118
x=0.5
m_host=x * MS #cgs
Mstar=m_host/ME  # convert to earthmasses
r_snow= x * (784/225) # snowline when T=150k
#Pstar= 319680
Pstar= 345680
rin_mig=(Grav * m_host * Pstar**2/(4 * np.pi**2))**(1/3) /au # corotation boundry in au
print(rin_mig)

# radial grid & disk setup (cgs)
Nr = 300  # number of grid points
Rin = 0.01*au
Rout = 1000.*au
rgrid = np.logspace(np.log10(Rin),np.log10(Rout),Nr)  #cm
msolids = 650.*ME * x # initial mass of solids #g
Z0 = 0.01  # solids-to-gas ratio
mdisk = msolids/Z0  # gas disk mass
rout = 30.*au  # critical radius of the disk
SigmaGas = mdisk / (2.*np.pi*rout**2.) * (rgrid/rout)**(-1.) * np.exp(-1.*(rgrid/rout))  # gas surface density
SigmaDust = Z0*SigmaGas  # dust surface density to go into pp only 
T = x**0.5 *280 * (rgrid/au)**(-0.5)  # temperature profile
alpha = 1.e-4  # turbulence strength parameter 
alphaz=alpha  # turbulance 
vfrag = 100 #if the snowline is turned on, vsnow=10*vfrag

# density variables
rhop = 1.25  # internal density of dust grains 
rhoi = 1.0  # density of ice g/cm^3
rhor = 3.0  # density of rock g/cm^3 - silicates and oxides
rhom= 8.0  # density of iron # metal and iron sulfide
rho = rhop/ME*au**3  # convert to human units 
rho_ice = rhoi/ME*au**3
rho_rock = rhor/ME*au**3
rho_metal = rhom/ME*au**3
rhoearth = 4.05 #g/cc

# timegrid setup
Nt = 1000
endtime = 1.e7*year  # s
timegrid = np.logspace(np.log10(year),np.log10(endtime),Nt)  # s

#check
if rin_mig <= Rin/au:
    sys.exit('inner boundary error')

#%%

# calling
planet_pop(initial_mass=1e-3)  # vfrag cgs others human units

winsound.Beep(500, 1000)  # frequency, duration
