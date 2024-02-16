import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import math
from tqdm.notebook import tqdm
from numba import njit

### Some notes:
# I had to remove all the plotting in the main function to get it to work with numba
# Some features from the original code aren't implemented. I didn't do depth-dependent properties because I didn't need to, but 
# adding this probably wouldn't be too hard
# Ice sublimation stuff is also commented out because I didn't need it and it was causing numba issues - should be a fairly quick fix though if needed


@njit
def iadd(a,b):
    length = min(len(a),len(b))
    return a[:length]+b[:length]
@njit
def isub(a,b):
    length = min(len(a),len(b))
    return a[:length]-b[:length]
@njit
def imult(a,b):
    length = min(len(a),len(b))
    return a[:length]*b[:length]
@njit
def idiv(a,b):
    length = min(len(a),len(b))
    return a[:length]/b[:length]
    
@njit
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

@njit
def dtherminit(zarr,rho,cp,k,emvty,dt):
    sigma = 5.67e-5 #erg cm-2 s-1 K-4
    emvtysigma = emvty*sigma
    wtmol = [0,16,28,18,44,28,44]
    nslab = len(zarr)
    notend = 1+np.arange(nslab - 2)
    endindex = nslab - 1
    zmid,zup,zdown,thick = depthset(zarr)
    koverz = imult(k,k[1:])/(imult(zup[1:],k)+imult(zdown,k[1:]))
    const = dt/(rho*cp*thick)
    
    return koverz,emvtysigma,notend,const,endindex,wtmol

@njit
def angsep(lat1,lon1,lat2,lon2):
    theta1 = np.pi/2-lat1
    theta2 = np.pi/2-lat2
    
    dphi = lon1-lon2
    arg = np.cos(theta1)*np.cos(theta2)+np.sin(theta1)*np.sin(theta2)*np.cos(dphi)
    arg = np.clip(arg,-1,1)
    angsep = np.arccos(arg)
    return angsep

@njit
def frostvap(t,ice=0):
# ICE=1 (solid methane), 2 (beta-solid nitrogen)
# ICE=3 (water ice) ICE=4 (CO2) from Bryson et al. 1980.
# ICE=5 (alpha-CO, from Brown and Zeigler)
# ICE=6 (CO2) from Brown and Zeigler.
# ICE=7 (SO2) from Wagman (1979)

    if ice==0:
        return 0
    if ice==7:
        frost_vap=1.516e8 * np.exp(-4510.0/t)*1e6
        return frost_vap

    A=[[1.71271658E1, -1.11039009E3, -4.34060967E3,  
        1.03503576E5, -7.91001903E5,  0.0         ], 
       [1.64302619E1, -6.50497257E2, -8.52432256E3,  
        1.55914234E5, -1.06368300E6,  0.0         ], 
       [21.7        , -5.74E3      ,  0.0         ,  
        0.0         ,  0.0         ,  0.0         ], 
       [23.8        , -3.27E3      ,  0.0         ,  
        0.0         ,  0.0         ,  0.0         ], 
       [1.80741183E1, -7.69842078E2, -1.21487759E4,  
        2.73500950E5, -2.90874670E6,  1.20319418E7], 
       [2.13807649E1, -2.57064700E3, -7.78129489E4,  
        4.32506256E6, -1.20671368E8,  1.34966306E9]]

    pmmhg = np.exp(A[0][ice-1]+A[1][ice-1]/t+A[2][ice-1]/t**2+A[3][ice-1]/t**3+A[4][ice-1]/t**4+A[5][ice-1]/t**5)
    frost_vap = pmmhg*1333.2
    return frost_vap

@njit
def frostheat(t,ice=0):
    # ICE=3 (H2O)
    # ICE=5 (alpha-CO, from Brown and Zeigler)
    # ICE=6 (CO2, Brown and Zeigler)   
    iceindex = ice-1
    A = [[2.18042941E03,  2.08564398E00, -1.25597469E-1, 8.20425270E-4, -2.78035132E-6,           0.0], 
       [1.74787117E03, -1.66883920E00,  2.80155636E-2,-5.52325058E-4,            0.0,           0.0], 
       [          0.0,            0.0,            0.0,           0.0,            0.0,           0.0], 
       [          0.0,            0.0,            0.0,           0.0,            0.0,           0.0], 
       [1.89343094E03,  7.33091373E00,  1.09658103E-2,-6.06576657E-3,  1.16607568E-4, -7.8957071E-7], 
       [6.26891952E03,  9.87685955E00, -1.30996632E-1, 6.27346551E-4, -1.26990433e-6,           0.0]]
    wtmol = [16,28,18,44,28,44]
    calht = A[0][iceindex]+A[1][iceindex]*t+A[2][iceindex]*t**2+A[3][iceindex]*t**3+A[4][iceindex]*t**4+A[5][iceindex]*t**5
    frost_heat = calht*4.18e7/wtmol[iceindex]
    if iceindex==2:
        frost_heat = 2838e7

    return frost_heat

@njit
def subrate(pvap,molwt,temp,stick=1):
    R = 8.31441e7
    subrate = stick*pvap*np.sqrt(1*molwt/(2*np.pi*R*temp))
    return subrate

@njit
def dtherm(temp,sol,koverz,emvtysigma,notend,const,endindex,wtmol,sundepth,frost=0):
    subcalc = 0
    if frost!=0:
        subcalc=1
            
    newtemp = temp
    # Conductive flow from slab below
    condflow = koverz*isub(temp[1:],temp)
    # Increment temperatures
    
    # Determine sublimation rate
    if subcalc==1:
        pvap = frostvap(temp[0],frost)
        subratet = subrate(pvap,wtmol[frost],temp[0],stick=1)
        subheat = frostheat(temp[0],frost)*subratet
    else:
        subheat=0
    # Only top slab can radiate:
    newtemp[0] = temp[0]+const[0]*(condflow[0]+sol[0]-emvtysigma*temp[0]**4-subheat)
    newtemp[notend] = temp[notend]+const[notend]*(condflow[notend]-condflow[notend-1]+sol[notend])
    #Uncomment for insulating lower boundary:
    if sundepth!=0:
        bottom = len(temp)-1
        newtemp[bottom] = temp[bottom]-const[bottom]*(condflow[bottom-1])
    return newtemp
@njit
def thermo_model(rhel,alb,ti,rot,emvty=1,rho=1,cp=1.5e7,plat=0,sunlat=0,
                 heatflow=0,sundepth=0,ice=0,insol=None,zarr=None,nrun=2,nslab=45,ntinc=5000,
                 nday=4,skdepths=9,showprof=True,showdiur=True,ectimes=np.array([-1,-1]),
                 corrfactor=0,blind=False,silent=False,waittime=1,noprompt=False,profwaittime=0,saveprofiles=True):
    
    r = rhel
    rotsec = rot*24*3600
#     try:
#         nslab = len(zarr)
#     except:
    nslab = nslab
        
#     try:
#         len(ti)
#         ti1 = ti
#     except:
    ti1 = ti*np.ones(nslab)
    
    
    k = ti1**2/rho/cp
    if not silent:
        print(' ')
        print('Albedo = ', alb, ', Upper layer TI = ', ti1[0], ', Heat flow = ', heatflow, 'erg cm-2 s-1')
        print('Thermal conductivity = ', k[0], ' erg cm-1 s-1 K-1')
    dt = rotsec / ntinc
    
   
    
    
    #Physical constants
    solk=1.374e6 
    sigma=5.670e-5
    
    #Skin depth
    skindepth = (k * rotsec / (2 * math.pi * rho * cp))**0.5
    if not silent:
        print('Skindepth for upper layer =', skindepth[0], ' cm')
#     try:
#         moddepth = max(zarr)
#     except:
    moddepth = skindepth[0] * skdepths
    
    #Thermophysical parameter theta
    theta = ti1[0] * (2 * math.pi / rotsec)**0.5 * rhel**1.5 / ((1 - alb)**0.75 * emvty**0.25 * sigma**0.25 * solk**0.75)
    if not silent:
        print('Theta =', theta)

    # Auto-determination of corrfac
    thetaarr = np.array([0.01, 0.10, 0.30, 1.01, 2.02, 3.03, 10.1, 30.4, 101.2, 1000.])
    corrfacarr = np.array([37., 37., 13., 3.80, 1.90, 1.30, 0.50, 0.33, 0.27, 0.27])
    if corrfactor==0:
        corrfactor = 10**(np.interp(math.log10(theta), np.log10(thetaarr),np.log10(corrfacarr)))
        if not silent:
            print('Corrfactor =', corrfactor)

    # Array initialization: Use double precision
#     try:
#         zarr = np.array([float(x) for x in zarr])
#     except:
    zarr = np.array([(i + 0.5) / (nslab - 0.5) * moddepth for i in range(nslab)])
    zarr0=zarr
    
    # Various parameters
    zmid,zup,zdown,thick = depthset(zarr)
    koverz,emvtysigma,notend,const,endindex,wtmol = dtherminit(zarr,rho,cp,k,emvty,dt)
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
#         netpower =0
#         if not silent:
#             print(sunlat / math.degrees(1), plat / math.degrees(1))
#         return 0,0,0,0,0,0,0# tsurf,tod,tdeep,teq,balance,tdarr,zarr0
    
    # Plot skindepth/slab thickness
    if showdiur:
#         fig,(ax1,ax2) = plt.subplots(1,2,figsize = (20,5))
#         ax1.plot(thick / skindepth, linestyle='None', marker='d', markersize=4)
#         ax1.set_ylabel('Slab thickness / Skindepth')
#         ax1.set_xlabel('Slab number')
#         ax1.set_title('Slab thickness report')
#         ax2.plot(zmid*10, linestyle='None', marker='d', markersize=4)
#         ax2.set_ylabel('Depth/mm')
#         ax2.set_xlabel('Slab number')
#         ax2.set_title('Slab depth report')
#         ax1.grid()
#         ax2.grid()
        
#         plt.show()
#         print('Average slab thickness: ' + str(np.mean(thick)) + 'cm')
        pass
        if not noprompt:
            pass
#             input("Press Enter to continue...")
            
    # Sunlight penetration
    sunfrac = np.zeros(nslab)
    if sundepth ==0:
        sunfrac[1:] = 0.0
        sunfrac[0] = 1.0
        sundepth = 0.0
    else:
        sunval = np.exp(-1*zmid/sundepth)
        sunfrac = sunval*thick
        sunfrac = sunfrac/np.sum(sunfrac)
        if not blind:
#             plt.figure(figsize=(10,5))
#             plt.plot(zmid, sunval/max(sunval))
#             plt.title("Sunlight penetration")
#             plt.xlabel("Depth, cm")
#             plt.ylabel("Normalized Solar Intensity")
#             plt.axvline(skindepth[0],c='r')
#             plt.text(skindepth[0], 0.95, "Skin Depth")
#             plt.axvline(sundepth, c='r')
#             plt.text(sundepth, 0.9, "Sun Depth")
#             plt.show()
            pass
            if not noprompt:
                pass
#                 input("Press Enter to continue...")
    
    # Total insolation each timestep
    if insol == None:
        suninc = angsep(plat,tod,sunlat,np.deg2rad(180))
        soltod = np.clip((1-alb)*solk/r**2*np.cos(suninc),0,1e10)
    else:
        suninc = angsep(plat,tod,sunlat,np.deg2rad(180))
        nsol = len(insol)
        soltod = (1 - alb) * np.interp(tod,np.arange(nsol + 1)/nsol*2*np.pi,np.append(insol, insol[0]))
#     plt.figure(figsize=(10,5))
#     plt.plot(tod,np.cos(suninc))
#     plt.show()

    # First guesses at surface and deep temps
    tsurf0 = ((0.25*np.pi*np.mean(soltod)+heatflow)/emvty/sigma)**0.25
    tdeep = tsurf0+heatflow*np.sum(thick/k)
    
    # Eclipse
    eclipse = np.where((tod>ectimes[0]*2*np.pi) & (tod<ectimes[1]*2*np.pi))[0]
    try:
        if eclipse[0]!=-1:
            soltod[eclipse]=0
    except:
        pass
    # Determine equilibrium temperatures
    teq = ((soltod+heatflow)/emvty/sigma)**0.25
    teqdark = (heatflow/emvty/sigma)**0.25
    if not silent:
        print('Equilibrium temperatures: Peak=',max(teq),', Dark=',teqdark)
    
    # Conductive stability criterion check
    stabcrit1a = idiv(koverz*dt,(thick*rho*cp))
    stabcrit1b = np.zeros(nslab)
    stabcrit1b[1:nslab-2]=koverz[0:nslab-3]*dt/(thick[1:nslab-2]*rho*cp)
    maxcrit = max([max(stabcrit1a),max(stabcrit1b)])
    if not silent:
        print('Maximum conductive stability criterion=',maxcrit)
    if maxcrit>0.5:
        print('Conductive stability violation! maxcrit=',maxcrit)
                                                                                        #ADD PLOTTING
        
    # Radiative stability criterion
#     try:
#         stabcrit2 = emvty*sigma*np.max(teq)**3*dt/(thick[0]*rho[0]*cp[0])
#     except:
    stabcrit2 = emvty*sigma*np.max(teq)**3*dt/(thick[0]*rho*cp)
    if not silent:
        print('Maximium radiative stability criterion=')
        print(stabcrit2)
    if stabcrit2>0.5:
        print('Radiative stability violation! stabcrit2=')
        print(stabcrit2)

    # Call dtherminit to set up parameters for integration
    koverz,emvtysigma,notend,const,endindex,wtmol = dtherminit(zarr,rho,cp,k,emvty,dt)
    
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
                                                                                         #ADD DAY PLOTTING
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
#         if ice!=0:
#             pvap = frostvap(tsurf,ice)
#             subratet = subrate(pvap,wtmol[ice],tsurf,stick=1)
#             subheat = frostheat(tsurf,ice)*subratet
        avein = np.sum(soltod)/ntinc
        aveout = (emvty*sigma*np.sum(tsurf**4)+subheat)/ntinc #replace with np.sum(subheat)
        tmean = np.sum(tsurf)/ntinc
        tdeepfinal = np.sum(tbase)/ntinc
        netpower = aveout-avein
        balance = (aveout-heatflow)/avein
        if not silent:
            print(irun,avein/1e3,aveout/1e3,balance,netpower/1e3,tmean,tdeepfinal)
        tdeep = tdeep/balance**corrfactor
        kmean = np.mean(k)
        if heatflow!=0:
            tsurf0 = tdeep-heatflow*moddepth/kmean
        else:
            tsurf0=tdeep
    tdeep = tdeepfinal
    return tsurf,tod,tdeep,teq,balance,tdarr,zarr0