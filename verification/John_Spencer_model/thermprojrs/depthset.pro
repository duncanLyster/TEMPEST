pro depthset,zarr,zmid,zup,zdown,thick
; Determines zmid,zup,zdown, and thick for an array of depths
; to the bottom of slabs, zarr.  Parameters defined as follows:
;
;  ------------------zmid(i)----------->
;                             <------thick(i)----->
;                             <--zup(i)-
;                                      -zdown(i)-->
;  ------------------zarr(i)---------------------->
;  Slab 0                            Slab i
;  reference                       reference
;  point at surface                  point
;  |          |               |        |          |
;  *----------+-------*-------+--------*----------+----
;  |   Slab 0 |               |    Slab i         |
; Noprompt, if set, supresses carriage return request


nslab=n_elements(zarr)
; zmid is the depth to the reference point for each slab
zmid=fltarr(nslab)
; For all but the top slab, the reference point is in the middle
zmid(1:*)=(zarr(1:*)+zarr(0:nslab-2))/2.0
; For the top slab, the reference point is on the surface
zmid(0)=0.0
; zdown is the distance from the reference point to the base
; of each slab
zdown=zarr-zmid
; zup is the distance from the reference point to the top of
; each slab
zup=fltarr(nslab)
zup(0)=zmid(0)
zup(1:*)=zmid(1:*)-zarr
; thick is the thickness of each slab, used for heat capacity
; and sunlight absorption
thick=fltarr(nslab)
thick(1:*)=zarr(1:nslab-1)-zarr
thick(0)=zarr(0)

return

end

