# SPY Stellar evolution in PYthon
# Test-bed for stellar turbulence models
# Evolved from STATSTAR

import STELLAR

Msolar = 1.   # mass of the star (in solar units)
Lsolar = 1.   # luminosity of the star (in solar units)
Te = 6000.    # effective temperature of the star (in K) 
X = 0.75      # mass fraction of hydrogen
Z = 0.02      # mass fraction of metals 
Nstart = 10
Nstop = 999
ierr = 0

star = STELLAR.spy(Msolar,Lsolar,Te,X,Z,Nstart,Nstop,ierr)

star.evol()
#star.store()
#star.plot()


