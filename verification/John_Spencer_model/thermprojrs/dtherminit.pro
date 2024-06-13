pro dtherminit,zarr,rho,c,k,emvty,dt
; Sets up common block so dtherm can increment temperatures in
; temperature/depth array (temp,zarr)
; over time dt, with thermophysical parameters rho, c, and k,
; which can be functions of zarr,
; and emissivity emvty,
; zarr is the distance to the bottom on each slab...

common dthermcommon,koverz,emvtysigma,notend,const,endindex,wtmol

sigma=5.670e-5  ; erg cm-2 s-1 K-4
emvtysigma=emvty*sigma

; molecular weights, for sublimation calculations if needed
;               CH4     N2    H2O    CO2   a-CO    CO2
wtmol =[   0,  16.0,  28.0,  18.0,  44.0,  28.0,  44.0]

nslab=n_elements(zarr)
; notend is an array of indices for all but the top and bottom
; slabs
notend=1+indgen(nslab-2)
endindex=nslab-1

; Define the various depth-related variables
depthset,zarr,zmid,zup,zdown,thick
; koverz is (conductivity)/(distance) to the slab below (notes uktherm/980226)
koverz=(k*k(1:*))/(zup(1:*)*k + zdown*k(1:*))

const=dt/(rho*c*thick)

end
