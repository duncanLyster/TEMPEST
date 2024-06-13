function subrate,pvap,molwt,temp,stick=stick
; Just returns 1-way sublimation rate, in g cm-2 s-1, from
; a frost with vapor pressure "pvap", dy cm-2 s-1, or 1e-6 bars,
; and molecular weight "molwt" g mol-1,
; at temperature "temp".  
; stick = sticking coefficient, default 1.0

if not keyword_set(stick) then stick=1.0

R=8.31441e7  ; erg mol-1 K-1

subrate=stick*pvap*sqrt(1.0*molwt/(2*!pi*R*temp))

return,subrate

end
