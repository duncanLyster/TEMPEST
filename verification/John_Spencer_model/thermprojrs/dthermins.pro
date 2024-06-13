function dthermins,temp,sol,frost=frost
; Increments temperatures in temperature/depth array (temp,z)
; over one timestep, with thermophysical parameters defined in
; dttherminit, q.v., and absorbed insolation sol (in energy flux units),
; which can be a function of z
; If frost is set and nonzero, includes sublimation of frost species: 1=CH4, 2=N2, 3=H2O, 5=alpha-CO, 6=CO2
; This version uses an insulating rather than a conductive lower boundary

; Get the parameters pre-calculated by dtherminit
common dthermcommon

subcalc=0
if keyword_set(frost) then if frost ne 0 then subcalc=1

newtemp=temp

; Condflow is the conductive heat flow from the slab below:
condflow=koverz*(temp(1:*)-temp)

; Increment the temperatures

; Determine sublimation rate, if applicable
if subcalc then begin
   pvap=frostvap(temp(0),frost)
   subratet=subrate(pvap,wtmol(frost),temp(0),stick=1)
   subheat=frostheat(temp(0),frost)*subratet
endif else subheat=0

; Only top slab can radiate:
newtemp(0)=temp(0) + const(0) * (  $
 condflow(0) + sol(0) - emvtysigma*temp(0)^4 - subheat)
newtemp(notend) = temp(notend) + const(notend) * (  $
 condflow(notend)-condflow(notend-1) + $
 sol(notend) )
bottom=n_elements(temp)-1
newtemp(bottom) = temp(bottom) - const(bottom) * ( $
 condflow(bottom-1) )

return,newtemp

end
