FUNCTION FROSTHEAT,T,ICE
; Determines cryogenic frost latent heat of sublimation in erg g-1 
; as a function of temperature T.  From Brown and Zeigler 1980 Adv. 
; Cryo. Eng. 25, 662-670. ICE=1 (solid methane), 2 (beta-solid nitrogen)
; ICE=3 (H2O)
; ICE=5 (alpha-CO, from Brown and Zeigler)
; ICE=6 (CO2, Brown and Zeigler)
; Translated from FORTRAN, 030403

iceindex=ice-1
a=[[2.18042941E03,  2.08564398E00, -1.25597469E-1, 8.20425270E-4, -2.78035132E-6,           0.0], $
   [1.74787117E03, -1.66883920E00,  2.80155636E-2,-5.52325058E-4,            0.0,           0.0], $
   [          0.0,            0.0,            0.0,           0.0,            0.0,           0.0], $
   [          0.0,            0.0,            0.0,           0.0,            0.0,           0.0], $
   [1.89343094E03,  7.33091373E00,  1.09658103E-2,-6.06576657E-3,  1.16607568E-4, -7.8957071E-7], $
   [6.26891952E03,  9.87685955E00, -1.30996632E-1, 6.27346551E-4, -1.26990433e-6,           0.0]]
WTMOL=[ 16.0, 28.0,18.0,44.0, 28.0, 44.0]
     
CALHT = A(0,iceindex) + A(1,iceindex)*T + A(2,iceindex)*T^2 + $
 A(3,iceindex)*T^3 + A(4,iceindex)*T^4 + A(5,iceindex)*T^5
FROSTHEAT=CALHT*4.18E7/WTMOL(iceindex)

; Special case for H2O where I only have the 0C latent heat
; (CRC handbook)
if iceindex eq 2 then frostheat=2838e7

RETURN,frostheat


END

