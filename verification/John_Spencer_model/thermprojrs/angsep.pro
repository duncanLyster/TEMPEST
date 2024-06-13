function angsep,lat1,lon1,lat2,lon2
; Determines angular separation of two directions on a sphere.
; lat=latitude, lon=longitude  
; Tested OK 5/28/87
; Translated from FORTRAN, 12/96, with colatitude changed to latitude

theta1=!pi/2-lat1
theta2=!pi/2-lat2

dphi=lon1-lon2
arg=cos(theta1)*cos(theta2)+sin(theta1)*sin(theta2)*cos(dphi)
arg=arg < 1
arg=arg > (-1)
angsep=acos(arg)

return,angsep

end



