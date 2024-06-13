FUNCTION FROSTVAP,T,ICE
; Determines cryogenic frost vapour pressure in dynes cm-2 (= microbars)
; as a function of temperature T.  
; From Brown and Zeigler 1980 Adv. Cryo. Eng. 25, 662-670.
; ICE=1 (solid methane), 2 (beta-solid nitrogen)
; ICE=3 (water ice) ICE=4 (CO2) from Bryson et al. 1980.
; ICE=5 (alpha-CO, from Brown and Zeigler)
; ICE=6 (CO2) from Brown and Zeigler.
; ICE=7 (SO2) from Wagman (1979)
; Converted from FORTRAN routine frostvap.f, 93/08/26
; Made array-friendly, 97/10/10
; SO2 added, 05/04/20

if n_elements(ice) eq 0 then begin
   print,'Usage: P / (dy cm-2) = frostvap(T, ice)'
   print,'Ice values: 1 = CH4; 2 = beta-N2; 3 = H2O; 4 = CO2; 5 = CO; 6 = CO2'
   return,0
endif

;print,'cie=',ice
if ice eq 7 then begin
;   print,'hi'
   frostvap=1.516e8 * exp(-4510.0/t)*1e6  ; Original formulation is in bars, so need 1e6 factor
   return,frostvap
endif

A=dblarr(6,6)
t=1d0*t
A=[[1.71271658E1, -1.11039009E3, -4.34060967E3,  $
    1.03503576E5, -7.91001903E5,  0.0         ], $
   [1.64302619E1, -6.50497257E2, -8.52432256E3,  $
    1.55914234E5, -1.06368300E6,  0.0         ], $
   [21.7        , -5.74E3      ,  0.0         ,  $
    0.0         ,  0.0         ,  0.0         ], $
   [23.8        , -3.27E3      ,  0.0         ,  $
    0.0         ,  0.0         ,  0.0         ], $
   [1.80741183E1, -7.69842078E2, -1.21487759E4,  $
    2.73500950E5, -2.90874670E6,  1.20319418E7], $
   [2.13807649E1, -2.57064700E3, -7.78129489E4,  $
    4.32506256E6, -1.20671368E8,  1.34966306E9]]
     
good=where(t ne 0)
bad=where(t le 0)
pmmhg=t
if good(0) ne -1 then $
 PMMHG(good) = EXP (A(0,ice-1) + A(1,ice-1)/T(good) + A(2,ice-1)/T(good)^2 +  $
 A(3,ice-1)/T(good)^3 + A(4,ice-1)/T(good)^4 + A(5,ice-1)/T(good)^5)
if bad(0) ne -1 then pmmhg(bad)=0
FROSTVAP = PMMHG * 1333.2

RETURN,frostvap

END

