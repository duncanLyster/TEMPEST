import numpy as np
import matplotlib.pyplot as plt
import math

def isub(a,b):
    length = min(len(a),len(b))
    return a[:length]-b[:length]

def imult(a,b):
    length = min(len(a),len(b))
    return a[:length]*b[:length]

def idiv(a,b):
    length = min(len(a),len(b))
    return a[:length]/b[:length]
    
# Define the depthset function, which calculates the midpoints, uppers, and lowers of the slabs
def depthset(zarr):
    nslab = len(zarr)
    zmid = np.zeros(nslab)
    zmid[1:] = (zarr[1:] + zarr[:nslab - 1]) / 2.0
    zmid[0] = 0.0
    zdown = zarr - zmid
    zup = np.zeros(nslab)
    zup[0] = zmid[0]
    zup[1:] = zmid[1:] - zarr[:nslab - 1]
    thick = np.zeros(nslab)
    thick[1:] = isub(zarr[1:],zarr)
    thick[0]=zarr[0]
       
    return zmid,zup,zdown,thick

# Define the dtherminit function, which initializes the parameters for the thermal model
def dtherminit(zarr,density,specific_heat_capacity,k,emisivity,dt):
    sigma = 5.67e-5 #erg cm-2 s-1 K-4
    emvtysigma = emisivity*sigma
    wtmol = [0,16,28,18,44,28,44]
    nslab = len(zarr)
    notend = 1+np.arange(nslab - 2)
    endindex = nslab - 1
    zmid,zup,zdown,thick = depthset(zarr)
    koverz = imult(k,k[1:])/(imult(zup[1:],k)+imult(zdown,k[1:]))
    const = dt/(density*specific_heat_capacity*thick)
    
    return koverz,emvtysigma,notend,const,endindex,wtmol

# Define the angular_separation function, which calculates the angular separation between two points
def angular_separation(lat1,lon1,lat2,lon2):
    theta1 = np.pi/2-lat1
    theta2 = np.pi/2-lat2
    
    dphi = lon1-lon2
    arg = np.cos(theta1)*np.cos(theta2)+np.sin(theta1)*np.sin(theta2)*np.cos(dphi)
    arg = np.clip(arg,-1,1)
    angular_separation = np.arccos(arg)
    return angular_separation

# Define the dtherm function, which calculates the temperature profile of the model
def dtherm(temp,sol,koverz,emvtysigma,notend,const,endindex,wtmol,sundepth,frost=0):
    subcalc = 0
    if frost!=0:
        subcalc=1
            
    newtemp = temp
    # Conductive flow from slab below
    condflow = koverz*isub(temp[1:],temp)
    # Increment temperatures
    
    subheat=0
    # Only top slab can radiate:
    newtemp[0] = temp[0]+const[0]*(condflow[0]+sol[0]-emvtysigma*temp[0]**4-subheat)
    newtemp[notend] = temp[notend]+const[notend]*(condflow[notend]-condflow[notend-1]+sol[notend])

    return newtemp

# Define the thermo_model function, which calculates the temperature profile of the model
def thermo_model(rhel,albedo,thermal_inertia,rot,emissivity=1,density=1,specific_heat_capacity=1.5e7,plat=0,sunlat=0,
                 heatflow=0,sundepth=0,ice=0,insol=None,zarr=None,nrun=2,nslab=45,ntinc=5000,
                 nday=4,skdepths=9,showprof=True,showdiur=True,ectimes=np.array([-1,-1]),
                 corrfactor=0,blind=False,silent=False,waittime=1,noprompt=False,profwaittime=0,saveprofiles=True):

    rotsec = rot*24*3600

    # Turn thermal inertia into a list
    thermal_inertias = thermal_inertia*np.ones(nslab)
    
    thermal_conductivities = thermal_inertias**2/density/specific_heat_capacity

    if not silent:
        print(' ')
        print('Albedo = ', albedo, ', Upper layer TI = ', thermal_inertias[0], ', Heat flow = ', heatflow, 'erg cm-2 s-1')
        print('Thermal conductivity = ', thermal_conductivities[0], ' erg cm-1 s-1 K-1')
    dt = rotsec / ntinc 
    
    # Physical constants
    solar_constant = 1.374e6
    sigma = 5.670e-5 # The Stefan-Boltzmann constant
    
    #Skin depth
    skindepth = (thermal_conductivities * rotsec / (2 * math.pi * density * specific_heat_capacity))**0.5

    if not silent:
        print('Skindepth for upper layer =', skindepth[0], ' cm')

    moddepth = skindepth[0] * skdepths
    
    #Thermophysical parameter theta
    theta = thermal_inertias[0] * (2 * math.pi / rotsec)**0.5 * rhel**1.5 / ((1 - albedo)**0.75 * emissivity**0.25 * sigma**0.25 * solar_constant**0.75)
    if not silent:
        print('Theta =', theta)

    # Auto-determination of corrfac
    thetaarr = np.array([0.01, 0.10, 0.30, 1.01, 2.02, 3.03, 10.1, 30.4, 101.2, 1000.]) # These values
    corrfacarr = np.array([37., 37., 13., 3.80, 1.90, 1.30, 0.50, 0.33, 0.27, 0.27])
    if corrfactor==0:
        corrfactor = 10**(np.interp(math.log10(theta), np.log10(thetaarr),np.log10(corrfacarr)))
        if not silent:
            print('Corrfactor =', corrfactor)

    # Array initialization: Use double precision
    zarr = np.array([(i + 0.5) / (nslab - 0.5) * moddepth for i in range(nslab)])
    zarr0=zarr
    
    # Various parameters
    zmid,zup,zdown,thick = depthset(zarr)
    koverz,emvtysigma,notend,const,endindex,wtmol = dtherminit(zarr,density,specific_heat_capacity,thermal_conductivities,emissivity,dt)
    temp = np.zeros(nslab)
    tsurf = np.zeros(ntinc)
    tbase = np.zeros(ntinc)
    sol = np.zeros(nslab)
    
    if saveprofiles:
        tdarr = np.zeros((nslab,200))
        
    # Time of day
    tod = np.array([i / ntinc * 2 * np.pi for i in range(ntinc)])
    
    # Return zeros for permanent darkness
    if insol == None and abs(sunlat - plat) > 0.999 * math.pi / 2:
        print('####### NO SUN - YOU WILL GET STRANGE RESULTS #######')

    # Sunlight penetration
    sunfrac = np.zeros(nslab)

    sunfrac[1:] = 0.0
    sunfrac[0] = 1.0
    sundepth = 0.0

    # Total insolation each timestep
    if insol == None:
        suninc = angular_separation(plat,tod,sunlat,np.deg2rad(180))
        soltod = np.clip((1-albedo)*solar_constant/rhel**2*np.cos(suninc),0,1e10)
    else:
        suninc = angular_separation(plat,tod,sunlat,np.deg2rad(180))
        nsol = len(insol)
        soltod = (1 - albedo) * np.interp(tod,np.arange(nsol + 1)/nsol*2*np.pi,np.append(insol, insol[0]))

    # First guesses at surface and deep temps
    tsurf0 = ((0.25*np.pi*np.mean(soltod)+heatflow)/emissivity/sigma)**0.25
    tdeep = tsurf0+heatflow*np.sum(thick/thermal_conductivities)
    
    # Eclipse
    eclipse = np.where((tod>ectimes[0]*2*np.pi) & (tod<ectimes[1]*2*np.pi))[0]
    try:
        if eclipse[0]!=-1:
            soltod[eclipse]=0
    except:
        pass

    # Determine equilibrium temperatures
    teq = ((soltod+heatflow)/emissivity/sigma)**0.25
    teqdark = (heatflow/emissivity/sigma)**0.25
    if not silent:
        print('Equilibrium temperatures: Peak=',max(teq),', Dark=',teqdark)
    
    # Conductive stability criterion check
    stabcrit1a = idiv(koverz*dt,(thick*density*specific_heat_capacity))
    stabcrit1b = np.zeros(nslab)
    stabcrit1b[1:nslab-2]=koverz[0:nslab-3]*dt/(thick[1:nslab-2]*density*specific_heat_capacity)
    maxcrit = max([max(stabcrit1a),max(stabcrit1b)])
    if not silent:
        print('Maximum conductive stability criterion=',maxcrit)
    if maxcrit>0.5:
        print('Conductive stability violation! maxcrit=',maxcrit)
        
    # Radiative stability criterion
    stabcrit2 = emissivity*sigma*np.max(teq)**3*dt/(thick[0]*density*specific_heat_capacity)

    if not silent:
        print('Maximium radiative stability criterion=')
        print(stabcrit2)
    if stabcrit2>0.5:
        print('Radiative stability violation! stabcrit2=')
        print(stabcrit2)

    # Call dtherminit to set up parameters for integration
    koverz,emvtysigma,notend,const,endindex,wtmol = dtherminit(zarr,density,specific_heat_capacity,thermal_conductivities,emissivity,dt)
    
    if not silent: 
        print('Energy Conservation Report at end of each run:')
        print('          Mean               Mean              Total             Net Power             Mean')
        print('        Power In           Power Out          Energy                Out                Surf.             Deep')
        print(' Run      W m-2              W m-2            In/Out               W m-2               Temp.             Temp.')
        print('--------------------------------------------------------------------------------------------------------------')
    
    # Energy conservation loop
    for irun in range(0,nrun):
        temp = tsurf0+np.arange(nslab)/(nslab-1)*(tdeep-tsurf0)
        
        for iday in range(0,nday):
        #initial temp profile is linear slope between estimated surface and deep temps
            # Timestep loop
            for it in range(0,ntinc):
                sol = soltod[it]*sunfrac

                temp = dtherm(temp,sol,koverz,emvtysigma,notend,const,endindex,wtmol,sundepth,frost=ice)
                tsurf[it] = temp[0]
                tbase[it] = temp[nslab-1]

                if saveprofiles and it%(ntinc/200)==0:
                    tdarr[:,int(it*200/ntinc)]=temp

        # Check energy conservation
        subheat = 0.
        
        avein = np.sum(soltod)/ntinc
        aveout = (emissivity*sigma*np.sum(tsurf**4)+subheat)/ntinc #replace with np.sum(subheat)
        tmean = np.sum(tsurf)/ntinc
        tdeepfinal = np.sum(tbase)/ntinc
        netpower = aveout-avein
        balance = (aveout-heatflow)/avein
        if not silent:
            print(irun,avein/1e3,aveout/1e3,balance,netpower/1e3,tmean,tdeepfinal)
        tdeep = tdeep/balance**corrfactor
        kmean = np.mean(thermal_conductivities)
        if heatflow!=0:
            tsurf0 = tdeep-heatflow*moddepth/kmean
        else:
            tsurf0=tdeep
    tdeep = tdeepfinal
    return tsurf,tod,tdeep,teq,balance,tdarr,zarr0

def main():
    tsurf,tod,tdeep,teq,balance,tdarr,zarr0 = thermo_model(rhel=9.0,albedo=0.5,rot=10.0,emissivity=0.9,thermal_inertia=1e4,plat=0.0,nrun=2,ntinc=5000) #,ectimes=np.array([0.5,0.6]))

    plt.figure()
    plt.plot(tod,tsurf)
    plt.xlabel('Rotation (rad)')
    plt.ylabel('Surface Temperature (K)')
    plt.show()

    # Save the temperature profiles
    print("Saving temperature profiles to 'temperature_profiles.csv'")
    np.savetxt('temperature_profiles.csv', tdarr, delimiter=',')

# Call the main program to start execution
if __name__ == "__main__":
    main()