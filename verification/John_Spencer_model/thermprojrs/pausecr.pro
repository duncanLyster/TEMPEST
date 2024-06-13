pro pausecr,nomessage=nomessage

; Pauses until a C/R is entered

if not keyword_set(nomessage) then $
 print,'Input C/R to continue ',format="($,a)"
buf='' & read,buf

return

end
