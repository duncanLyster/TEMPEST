pro thermprojrs,tsurf,tod,tdeep,teq,balance,tdarr,zarr0, $
; 1-dimensional surface thermal model for a rotating body (arbitrary
; insolation dependence with time is also catered for)

; Output variables (Position-dependent: later items in the list can be omitted):
;   tsurf:    diurnal surface temperature array
;   tod:      time of day array (in radians)
;   tdeep:    final deep temperature
;   teq:      Instantaneous-equilibrium temperature array
;   balance:  Thermal balance: (power out - heatflow)/(power in)
;   tdarr:    temperature/depth array, with 200 diurnal increments
;             (set this to a defined variable before entering thermprojrs).
;   zarr0:    for convenience, returns the array of slab-bottom depths
;              (same as zarr)

; Required physical keyword parameters
rhel=rhel,         $ ; heliocentric distance, AU
alb=alb,           $ ; bolometric albedo
ti=ti,             $ ; thermal inertia, erg-cgs.  Can be an array, in
;                      which case it gives the ti of each slab
rot=rot,           $ ; Rotation period, days
; Optional physical keyword parameters:
emvty=emvty,       $ ; emissivity                   (default 1.0)
rho=rho,           $ ; density, g cm-3.  Can be an array, giving the density of each slab. (default 1.0)
cp=cp,             $ ; specific heat, erg g-1 K-1 (default H20)
plat=plat,         $ ; latitude (radians)           (default 0.0)
sunlat=sunlat,     $ ; Subsolar latitude (radians)  (default 0.0)
heatflow=heatflow, $ ; Endogenic heat flow, erg cm-2 s-1 (default 0.0)
ectimes=ectimes,   $ ; 2-element array of eclipse start and end times, in fractional days (default none)
sundepth=sundepth, $ ; e-folding depth for sunlight penetration, cm (default 0.0)
ice=ice,           $ ; Ice species (1=CH4, 2=N2, 3=H2O, 5=alpha-CO, 6=CO2), for inclusion of sublimation (default 0: no ice)
insol=insol,       $ ; If set, describes an arbitrary function of insolation vs. time through a 
                     ; full cycle of length "rot" days, in erg cm-2 s-1, with constant time spacing.  Insolation 
                     ; at the end of the cycle should not be specified, and is assumed to
                     ; equal the insolation at the beginning of the cycle. 
; Optional simulation keyword parameters:
zarr=zarr,         $ ; Array of depths to the BASE of each slab (NOT slab thicknesses), in cm.  First element must be non-zero.
nrun=nrun,         $ ; Number for runs for energy balance           (default 2)
nslab=nslab,       $ ; Number of slabs (ignored if zarr set)        (default 30)
ntinc=ntinc,       $ ; Time increments per day                      (default 5000)
nday=nday,         $ ; Number of days per run                       (default 4)
skdepths=skdepths, $ ; Model depth, in skindepths (ignored if zarr set)  (default 6)
corrfactor=corrfactor, $ ; Factor for convergence of deep Ts on successive runs (notes 980217)
;                          Larger values cause larger adjustments of deep T (default 0.5)
showprof=showprof, $ ; Show T profiles with depth                           (default yes)
showdiur=showdiur, $ ; Show diurnal T profiles after each run               (default yes)
blind=blind,       $ ; Don't display anything or stop for any prompts
noprompt=noprompt, $ ; Don't stop for prompts
waittime=waittime, $ ; Time to wait after each run, seconds (default 1.0)
profwaittime=profwaittime, $  ; Time to wait after each plotted timestep final depth profile, seconds (default 0.0)
silent=silent        ; If set, sends no output to the terminal (except stability warnings)
;
; EXAMPLES:

; ; Simple homogeneous model:
; thermprojrs,tsurf,tod,rhel=9.0,alb=0.5,rot=10.0,emvty=0.9,ti=1e4
; plot,tod/!dtor,tsurf
;
; ; Depth-dependent parameters: low-thermal-inertia layer over
; ; high-thermal inertia layer.  Time resolution increased over default
; ; value to maintain numerical stability:
; tiarr=[1e4+indgen(5),1e5+indgen(20)]
; zarr=[0.02*(1+indgen(5)),1+0.1*(1+indgen(20))]
; thermprojrs,tsurf,tod,rhel=9.0,alb=0.5,rot=10.0,emvty=0.9,ti=tiarr,zarr=zarr,ntinc=10000
; plot,tod/!dtor,tsurf
;
; NOTES:
; Conductivity is not specified, but derived from thermal inertia, specific heat,
; and density
; Doesn't handle situations of permanent darkness (|(latitude-sslatitude)| > 90 degrees)
; very well.
;
; See readme.txt for more details
;
; Report bugs or suggestions for improvement to:
; John Spencer, Southwest Research Institute, Boulder
; spencer@boulder.swri.edu

; March 11th 2005: first "published" version
; April 5th 2005: Added "silent" option, replaced "netpower" output
;                 variable with more useful "balance" variable
; July 19th 2005: Added arbitrary insolation variation option

; Get access to the parameters pre-calculated by dtherminit
common dthermcommon,koverz,emvtysigma,notend,const,endindex,wtmol

; Defaults
if not keyword_set(emvty) then emvty=1.0
if not keyword_set(rho) then rho=1.0
;if not keyword_set(cp) then cp=8e6   ; H2O at 90 K from Spencer and Moore 1992
if not keyword_set(cp) then cp=1.5e7   ; Basalt flows, from Davies 1996
if not keyword_set(plat) then plat=0.0
if not keyword_set(sunlat) then sunlat=0.0
if not keyword_set(heatflow) then heatflow=0.0
if not keyword_set(nrun) then nrun=10
if not keyword_set(nslab) and not keyword_set(zarr) then nslab=45 $
 else if keyword_set(zarr) then nslab=n_elements(zarr)
if not keyword_set(ntinc) then ntinc=5000
if not keyword_set(nday) then nday=10
if not keyword_set(skdepths) then skdepths=9
if not keyword_set(ectimes) then ectimes=[-1,-1]
;if not keyword_set(corrfactor) then corrfactor=0.5
if not keyword_set(blind) then blind=0
if not blind then begin
   if n_elements(showprof) eq 0 then showprof=1
   if n_elements(showdiur) eq 0 then showdiur=1
endif else begin
   showprof=0
   showdiur=0
endelse
if not keyword_set(silent) then silent=0

if n_elements(waittime) eq 0 then waittime=1.0
if not keyword_set(profwaittime) then profwaittime=0.0
saveprofiles= n_elements(tdarr) ne 0

; Make ti (and thus k) into an array even if it is constant with
; depth, for computational simplicity.  Rename the variable to
; ti1 so the input parameter isn't altered by thermpro
if n_elements(ti) eq 1 then ti1=replicate(ti,nslab) else ti1=ti

r=rhel
rotsec=rot*24.0*3600
k=ti1^2/rho/cp
if not silent then print,' '
if not silent then print,'Albedo=',alb,',  Upper layer TI=',ti(0),',  Heat flow=',heatflow,'erg cm-2 s-1', $
 format="(a,f5.2,a,e10.2,a,e10.2,a)"
if not silent then print,'Thermal conductivity=',k(0),' erg cm-1 s-1 K-1',format="(a,e10.2,a)"
dt=rotsec/ntinc

; Physical constants:
solk=1.374e6 ; solar constant, Hanel et al., erg cm-2 s-1
sigma=5.670e-5  ; Stefan/Boltzmann, erg cm-2 s-1 K-4

; Print out factors that go into skindepth calculation
if not silent then print,'Factors in skindepth calculation:',format="(a)"
if not silent then print,'  k=',k(0),',  rotsec=',rotsec,',  rho=',rho(0),',  cp=',cp(0),format="(5f10.2)"

; Print out factors going into k
if not silent then print,'Factors in k calculation:',format="(a)"
if not silent then print,'  ti=',ti1,',  rho=',rho,',  cp=',cp,format="(3f10.2)"

; Skindepth
skindepth=sqrt(k*rotsec/(2*!pi*rho*cp))
; Print out skindepth and model depth
if not silent then print,'Skindepth for upper layer =',skindepth(0),' cm',format="(a,f8.3,a)"
if keyword_set(zarr) then moddepth=max(zarr) else $
 moddepth=skindepth(0)*skdepths
 thick=moddepth/nslab
; Print out model depth and number of slabs
if not silent then print,'Model depth=',moddepth,' cm',format="(a,f8.3,a)"
if not silent then print,'Number of slabs=',nslab,format="(a,i3,a)"
print,'Slab thicknesses:',thick,format="(a,10f8.3)"

; Thermophysical parameter Theta (Spencer et al. 1989)
theta=ti1(0)*sqrt(2*!pi/rotsec)*rhel^1.5 / ((1-alb)^0.75*emvty^0.25*sigma^0.25*solk^0.75)
if not silent then print,'Theta=',theta,format="(a,f7.3)" 

; Auto-determination of corrfac- see notes 050408
thetaarr=  [0.01,0.10,0.30,1.01,2.02,3.03,10.1,30.4,101.2,1000.]
corrfacarr=[37., 37., 13.,3.80,1.90,1.30,0.50,0.33, 0.27, 0.27]
if not keyword_set(corrfactor) then corrfactor=10^(interpol(alog10(corrfacarr),alog10(thetaarr),alog10(theta)))

; Array initialization: Use double precision (notes 980217)
if keyword_set(zarr) then zarr=double(zarr) else $
 zarr=(dindgen(nslab)+0.5)/(nslab-0.5)*moddepth
zarr0=zarr

; Define the various depth-related variables
if keyword_set(noprompt) then promptval=1 else promptval=0
depthset,zarr,zmid,zup,zdown,thick
; koverz is (conductivity)/(distance) to the slab below (notes 980226)
koverz=(k*k(1:*))/(zup(1:*)*k + zdown*k(1:*))

temp=dblarr(nslab)
tsurf=dblarr(ntinc)
tbase=dblarr(ntinc)
sol=dblarr(nslab) & sol(*)=0
if saveprofiles then tdarr=fltarr(nslab,200)
; Time of day each timestep, radians from midnight
tod=dindgen(ntinc)/ntinc*2*!pi

; Return zeros for situations with permanent darkness
if not keyword_set(insol) and abs(sunlat-plat) gt 0.999*!pi/2 then begin
   netpower=0
   if not silent then print,sunlat/!dtor,plat/!dtor
   goto,final
endif

; Plot skindepth /(slab thickness)
if showdiur then begin
   plot,thick/skindepth,psym=-4,ytitle='Slab thickness / Skindepth', $
    xtitle='Slab number',title='Slab thickness report'
   if not keyword_set(noprompt) then pausecr
endif

; Sunlight penetration
sunfrac=fltarr(nslab)
if not keyword_set(sundepth) then begin
   sunfrac(1:*)=0.0
   sunfrac(0)=1.0
   sundepth=0.0
endif else begin
   sunval=exp(-1*zmid/sundepth)
   sunfrac=sunval*thick
   sunfrac=sunfrac/total(sunfrac)  ; Normalize
   if not blind then begin
      plot,zmid,sunval/max(sunval),title='Sunlight penetration', $
       xtitle='Depth, cm',ytitle='Normalized Solar Intensity',charsize=1.2
      oplot,skindepth+[0,0],!y.crange,linestyle=1
      xyouts,skindepth,0.95,'Skin Depth'
      oplot,sundepth+[0,0],!y.crange,linestyle=1
      xyouts,sundepth,0.9,'Sun Depth'
      if not keyword_set(noprompt) then pausecr
   endif
endelse

; Total insolation each timestep
if not keyword_set(insol) then begin
   suninc=angsep(plat,tod,sunlat,180*!dtor)
   soltod=(1-alb)*solk/r^2*cos(suninc) > 0
endif else begin
   nsol=n_elements(insol)
   soltod=(1-alb)*interpol([insol,insol(0)],findgen(nsol+1)/(nsol)*2*!pi,tod)
endelse
;plot,tod,cos(suninc)
;pausecr

; Determine first guesses at surface and deep temperatures
; (factor of 0.25*!pi is empirical accounting for diurnal averaging)
tsurf0=((0.25*!pi*mean(soltod)+heatflow)/emvty/sigma)^0.25
tdeep=tsurf0+heatflow*total(thick/k)

; Eclipse (instantaneous start and end)
eclipse=where(tod gt ectimes(0)*2*!pi and tod lt ectimes(1)*2*!pi)
if eclipse(0) ne -1 then soltod(eclipse)=0.0


; Determine equilibrium temperatures, including internal heat
teq=((soltod+heatflow)/emvty/sigma)^0.25
teqdark=(heatflow/emvty/sigma)^0.25
if not silent then print,'Equilibrium temperatures: Peak=',max(teq),',  Dark=',teqdark, $
 format="(a,f8.2,a,f8.2)"


; Conductive stability criterion check (put all this in dtherminit eventually?):
stabcrit1a=koverz*dt/(thick*rho*cp)   ; To layer below
stabcrit1b=fltarr(nslab)
stabcrit1b(1:nslab-2)=koverz(0:nslab-3)*dt/(thick(1:nslab-2)*rho*cp)   ; To layer above
maxcrit=max([stabcrit1a,stabcrit1b])
if not silent then print,'Maximum conductive stability criterion=',maxcrit,format="(a,f8.3)"
if maxcrit gt 0.5 then begin
   print,'thermprojrs: Conductive stability violation!! maxcrit=',maxcrit,format="(a,f8.3)"
   if not blind then begin
      plot,indgen(nslab)-0.5,stabcrit1b,psym=-1,symsize=0.5, $
       yrange=[min([stabcrit1a,stabcrit1b]),maxcrit], $
       linestyle=2,ytitle='Conductive stability criterion (should be <0.5)', $
       xtitle='Slab',title='Conductive stability violation!!'
      oplot,indgen(nslab)+0.5,stabcrit1a,psym=-4,symsize=0.5
      oplot,!x.crange,[0.5,0.5]
      if not keyword_set(noprompt) then pausecr
   endif
endif
; Radiative stability criterion check (top slab only):
stabcrit2=emvty*sigma*max(teq)^3*dt/(thick(0)*rho(0)*cp(0))
if not silent then print,'Maximum Radiative stability criterion =',stabcrit2,format="(a,f8.3)"
if stabcrit2 gt 0.5 then begin
   print,'thermprojrs: Radiative stability violation!!  stabcrit2=',stabcrit2,format="(a,f8.3)"
   if not blind and not keyword_set(noprompt) then pausecr
endif
if not silent then print,''

; Call dtherminit to set up parameters for the integration
dtherminit,zarr,rho,cp,k,emvty,dt

if not silent then print,'Energy Conservation Report at end of each run:'
if not silent then print,'        Mean       Mean      Total  Net Power  Mean'
if not silent then print,'      Power In   Power Out  Energy     Out     Surf.  Deep'
if not silent then print,' Run    W m-2      W m-2    In/Out    W m-2    Temp.  Temp.'
if not silent then print,'-----------------------------------------------------------'

; Energy conservation loop:
for irun=0,nrun-1 do begin
; Initial temperature profile is a linear slope between the
; estimated surface and deep temperatures
   temp=tsurf0+dindgen(nslab)/(nslab-1)*(tdeep-tsurf0)
;   print,'tdeep, tsurf0',tdeep,tsurf0
; Day loop:
   for iday=0,nday-1 do begin
      if showprof then begin $
         plot,[0],/nodata,xrange=[0,max(zmid)],yrange=[min(tsurf),(max(tsurf)+1>tdeep)], $
          xtitle='Depth, cm',ytitle='T, K',title='Temperature Profile with Depth, '+  $
          'Run '+string(irun,format="(i3)")+'  Day '+string(iday,format="(i3)"),charsize=1.2
         oplot,skindepth+[0,0],!y.crange,linestyle=1
         xyouts,skindepth,!y.crange(0)+0.95*(!y.crange(1)-!y.crange(0)),'Skin Depth'
         oplot,sundepth+[0,0],!y.crange,linestyle=1
         xyouts,sundepth,!y.crange(0)+0.9*(!y.crange(1)-!y.crange(0)),'Sun Depth'
      endif
; Timestep loop:
      for it=long(0),ntinc-1 do begin
         sol=soltod(it)*sunfrac
; Uncomment next statement for conducting lower boundary
         if not keyword_set(sundepth) then $
          temp=dtherm(temp,sol,frost=ice) else $
          temp=dthermins(temp,sol)
; Uncomment next line for insulating lower boundary
;         temp=dthermins(temp,sol)
         tsurf(it)=temp(0)
         tbase(it)=temp(nslab-1)
         ; Print out temperature vs. depth for first 20 eclipse steps
         ;if it ge eclipse(0)-2 and it le (eclipse(0)+20) and iday eq 0 $
         ; then print,temp(0:6)
         if showprof and it mod (ntinc/16) eq 0 then begin
            oplot,zmid,temp,psym=-1,symsize=0.5
         endif
         if saveprofiles and it mod (ntinc/200) eq 0 then tdarr(*,it*200/ntinc)=temp
      endfor
      if showprof and iday eq nday-1 and it gt 14*ntinc/16 then wait,profwaittime
;      pausecr
      if showdiur then begin
         plot,[tod,tod+2*!pi]/!dtor,[tsurf,tsurf],/ynozero, $
          xrange=[0,720],xstyle=1,xticks=8,xminor=3,xticklen=1, $
          title='Diurnal Surface Temperatures, Run '+string(irun,format="(i3)")+ $
          '  Day '+string(iday,format="(i3)"),xtitle='Time of Day (repeated)',ytitle='T, K'
         oplot,[tod,tod+2*!pi]/!dtor,[teq,teq],linestyle=1
         oplot,interpol(!x.crange,[0,1],[0.4,0.49]),interpol(!y.crange,[0,1],0.9+[0,0]),linestyle=1
         xyouts,interpol(!x.crange,[0,1],0.505),interpol(!y.crange,[0,1],0.895),'Equilibrium T'
         if showprof then wait,waittime
;         pausecr
      endif
   endfor

; Check energy conservation:
   subheat=0
   if keyword_set(ice) then begin
      if ice ne 0 then begin
         pvap=frostvap(tsurf,ice)
         subratet=subrate(pvap,wtmol(ice),tsurf,stick=1)
         subheat=frostheat(tsurf,ice)*subratet
      endif
   endif
   avein=total(soltod)/ntinc
   aveout=(emvty*sigma*total(tsurf^4)+total(subheat))/ntinc
   tmean=total(tsurf)/ntinc
   tdeepfinal=total(tbase)/ntinc
   netpower=aveout-avein
   balance=(aveout-heatflow)/avein
   if not silent then print,irun,avein/1e3,aveout/1e3,balance,netpower/1e3,tmean,tdeepfinal,format="(i4,e11.3,e11.3,f8.4,e11.3,f7.2,f7.2)"

;   tdeep=tdeep*((avein+heatflow)/aveout)^corrfactor
   tdeep=tdeep/balance^corrfactor
;   print,'New deep T=',tdeep,format="(a,f7.2)"
; Still not sure about best way to set up next temperature profile: see notes.txt,
; 970408
   kmean=total(k)/n_elements(k)
   if heatflow ne 0 then $
    tsurf0=tdeep-heatflow*moddepth/kmean $
    else tsurf0=tdeep

; Save the final results to a CSV file
openw, lun, 'final_output_data.csv', /get_lun
; Write the header
printf, lun, 'Time of Day (radians), Surface Temperature (K)'
; Write the data
for i=0, n_elements(tod) - 1 do begin
   printf, lun, format = '(F10.5, ",", F10.5)', tod[i], tsurf[i]
endfor
close, lun
endfor

tdeep=tdeepfinal

final:

return

end
