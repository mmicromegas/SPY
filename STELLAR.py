# class for the SPY code (evolved from STATSTAR)


import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import pandas as pd

class spy:

    # define class constants/variables
    
    sigma = 5.67051e-5  # Stefan-Boltzmann constant
    c = 2.99792458e10   # speed of light in vacuum
    a = 7.56591e-15     # 4*sigma/c radiation pressure constant
    G =  6.67259e-8     # gravitational constant
    k_B = 1.380658e-16  # Boltzmann constant
    m_H = 1.673534e-24  # mass of hydrogen atom
    gamma = 1.6666667   # 5./3. adiabatic gamma for a monoatomic gas
    gamrat = gamma/(gamma-1.)
#    tog_bf = 0.01       # bound-free opacity constant (guillotine/gaunt)
    g_ff = 1.0          # free-free opactity gaunt factor (assummed 1)
    Rsun = 6.9599e10    
    Msun = 1.989e33   
    Lsun = 3.826e33
    Igoof = -1 # number of zones exceeded
    
    def __init__(self,Msolar,Lsolar,Te,X,Z,Nstart,Nstop,ierr):
        self.Msolar = Msolar
        self.Lsolar = Lsolar
        self.Te = Te
        self.X = X
        self.Z = Z
        self.Y = 1.-X-Z  # calculate fraction of helium
        self.XCNO = Z/2. # select the mass fraction of CNO to be 0.5 of Z
        self.Nstart = Nstart
        self.Nstop = Nstop
        self.ierr = ierr
        self.gamrat = self.gamma/(self.gamma-1.0)

    def evol(self):
        r = []
        P = []
        M_r = []
        L_r = []
        T = []
        rho = []
        kappa = []
        epsilon = []
        dlPdlT  = []

        Ms = self.Msolar*self.Msun
        Ls = self.Lsolar*self.Lsun
        Rs = np.sqrt(Ls/(4.*np.pi*self.sigma))/self.Te**2
        Rsolar = Rs/self.Rsun

        deltar = -Rs/1000.
        idrflg = 0
        Igoof = -1
        dlPlim = 99.
        
        mu = 1.0/(2.*self.X+0.75*self.Y+0.5*self.Z)
        tog_bf = 0.01
        
#        gamrat = self.gamma/(self.gamma-1.0)

        T0 = 0.
        P0 = 0.
        
        r.append(Rs)
        M_r.append(Ms)
        L_r.append(Ls)
        T.append(T0)
        P.append(P0)

        if (P0 <= 0.0 or T0 <= 0.0):
            rho.append(0.0)
            kappa.append(0.0)
            epsilon.append(0.0)
        else:
            sol = self.EOS(mu,P[0],T[0],0)
            rho.append(sol[0])
            kappa.append(sol[1])
            epsilon.append(sol[2])
    
        kPad = 0.3
        irc = 0
        dlPdlT.append(4.25)

        cnstmass = True
        for i in range(self.Nstart):
            if cnstmass: 
                ip1 = i + 1
                smdl = self.STARTMDL(deltar,mu,Rs,r[i],M_r[i], \
                                     L_r[i],tog_bf,irc)
            
                r.append(smdl[0])
                P.append(smdl[1])
                M_r.append(smdl[2])
                L_r.append(smdl[3])
                T.append(smdl[4])
            
                eossol = self.EOS(mu,P[ip1],T[ip1],ip1)

                rho.append(eossol[0])
                kappa.append(eossol[1])
                epsilon.append(eossol[2])
                tog_bf = eossol[3]


                print(i,rho[i],T[i],P[i])
                
                
                ierr = eossol[4]
                if (ierr != 0):
                    print("Error in EOS during STARTMDL: ",i,r[i]/Rs,rho[i], \
                          M_r[i]/Ms,kappa[i],T[i],epsilon[i],P[i],L_r[i]/Ls)
                    sys.exit()

                if (i > 0):
                    dlPdlTloc = np.log(P[ip1]/P[i])/np.log(T[ip1]/T[i])
                    dlPdlT.append(dlPdlTloc)
                else:
                    dlPdlT.append(dlPdlT[i])

                if (dlPdlT[ip1] < self.gamrat):
                    irc = 1
                else:
                    irc = 0
                    kPad = P[ip1]/T[ip1]**self.gamrat # check the formula

            deltaM = deltar*self.dMdr(r[ip1],rho[ip1])
            M_r[ip1] = M_r[i] + deltaM
            if (np.abs(deltaM) > 0.001*Ms):
                print("The variation in mass has become larger than 0.001*Ms \
                   leaving the approximation loop before Nstart was reached")
                if (ip1 > 1):
                    ip1 = ip1 - 1
                    cnstmass = False
                    
        Nsrtp1 = ip1 + 1
        for i in range(Nsrtp1,self.Nstop):
            if (Igoof == -1):
                im1 = i - 1
                f_im1 = []
                dfdr  = []
                f_im1.append(P[im1])
                f_im1.append(M_r[im1])
                f_im1.append(L_r[im1])
                f_im1.append(T[im1])
                dfdr.append(self.dPdr(r[im1],M_r[im1],rho[im1]))
                dfdr.append(self.dMdr(r[im1],rho[im1]))
                dfdr.append(self.dLdr(r[im1],rho[im1],epsilon[im1]))
                dfdr.append(self.dTdr(r[im1],M_r[im1], L_r[im1], T[im1],\
                                      rho[im1],kappa[im1],mu,irc))
                rgsol = self.RUNGE(f_im1,dfdr,r[im1],deltar,irc,mu,i)

                f_i  = rgsol[0]
                ierr = rgsol[1]
                if (ierr != 0):
                    print("The problem occurred in the Runge-Kutta routine")
                    print(r[im1]/Rs, rho[im1],M_r[im1]/Ms,kappa[im1],\
                          T[im1],epsilon[im1],P[im1],L_r[im1]/Ls)

                r.append(r[im1]+deltar)
                P.append(f_i[0])
                M_r.append(f_i[1])
                L_r.append(f_i[2])
                T.append(f_i[3])

                eossol = self.EOS(mu,P[i],T[i],i)

                rho.append(eossol[0])
                kappa.append(eossol[1])
                epsilon.append(eossol[2]) 
            
                ierr = eossol[4]
                if (ierr != 0):
                    print(r[im1]/Rs,rho[im1],M_r[im1]/Ms,kappa[im1],\
                          T[im1],epsilon[im1],P[im1],L_r[im1]/Ls)
                    sys.exit()

                dlPdlT.append(np.log(P[i]/P[im1])/np.log(T[i]/T[im1]))
                if (dlPdlT[i] < self.gamrat):
                    irc = 1
                else:
                    irc = 0

                if (r[i] <= np.abs(deltar) and (L_r[i] >=0.1*Ls or \
                                                M_r[i]>=0.01*Ms)):
                    Igoof = 6
                elif (L_r[i] <= 0.0):
                    Igoof = 5
                    rhocor = M_r[i]/((4./3.)*np.pi*r[i]**3)
                    if (M_r[i] != 0.0):
                        epscor = L_r[i]/M_r[i]
                    else:
                        epscor = 0.0
                    Pcore = P[i] + 2.0/3.0*np.pi*self.G*rhocor**2*r[i]**2
                    Tcore = Pcore*mu*self.m_H/(rhocor*self.k_B)
                elif (M_r[i] <= 0.0):
                    Igoof = 4
                    Rhocor = 0.0
                    epscor = 0.0
                    Pcore = 0.0
                    Tcore = 0.0					
                elif (r[i]<0.02*Rs and M_r[i]<0.01*Ms and L_r[i]<0.1*Ls):
                    rhocor = M_r[i]/((4./3.)*np.pi*r[i]**3)
                    rhomax = 10.*(rho[i]/rho[im1])*rho[i]
                    epscor = L_r[i]/M_r[i]
                    Pcore = P[i] + 2.0/3.0*np.pi*self.G*rhocor**2*r[i]**2
                    Tcore = Pcore*mu*self.m_H/(rhocor*self.k_B)
                    if (rhocor < rho[i] or rhocor > rhomax):
                        Igoof = 1
                    elif (epscor < epsilon[i]):
                        Igoof = 2
                    elif (Tcore < T[i]):
                        Igoof = 3
                    else:
                        Igoof = 0

                if (idrflg == 0 and M_r[i] < 0.99*Ms):
                    deltar = -Rs/100.
                    idrflg = 1
                if (idrflg == 1 and deltar >= 0.5*r[i]):
                    deltar = -Rs/5000.
                    idrflg = 2
                istop = i
                if (i % 1) == 0:
#                    print(i,deltar)
		    print(i,rho[i],T[i],P[i],L_r[i], Igoof)
#                    print(i,deltar,round(r[i],1),round(M_r[i],1),round(L_r[i],1),round(r[i]/Rs,1),round(M_r[i]/Ms,1),round(L_r[i]/Ls,1))
            
# generate warning messages for the central conditions

        rhocor = M_r[istop]/(4.0/3.0*np.pi*r[istop]**3)
        epscor = L_r[istop]/M_r[istop]
        Pcore  = P[istop] + 2.0/3.0*np.pi*self.G*rhocor**2*r[istop]**2
        Tcore  = Pcore*mu*self.m_H/(rhocor*self.k_B)

        print('Igoof: ', Igoof)
        if (Igoof != 0):
            if (Igoof == -1):
                print("Sorry to be the bearer of bad news, but your model \
                       has some problems.")
                print("The number of allowed shells has been exceeded.")
            elif (Igoof == 1):
                print("It looks like you are getting close, however there are \
                       still a few minor errors.")
                print("The core density seems a bit off, density should \
                       increase smoothly toward the center. The density \
                       of the last zone calculated was rho = ",rho[istop])
                if (rhocor > 1.e10):
                    print("It looks like you will need a degenerate neutron \
                           gas and general relativity to solve this core")
            elif (Igoof == 2):
                print("It looks like you are getting close, however there are \
                       still a few minor errors.")
                print("The core epsilon seems a bit off, epsilon should \
                           vary smoothly near the center. The value calculated \
                           for the last zone was eps = ",epsilon[istop])
            elif (Igoof == 3):
                print("It looks like you are getting close, however there are \
                       still a few minor errors")
                print("Your extrapolated central temperature is too low, \
                       a little more fine tuning ought to do it. The value \
                       calculated for the last zone was T = ",T[istop])
            elif (Igoof == 4):
                print("Sorry to be the bearer of bad news, but your model \
                       has some problems.")
                print("You created a star with a hole in the center!")
            elif (Igoof == 5):
                print("Sorry to be the bearer of bad news, but your model \
                       has some problems.")
                print("This star has a negative central luminosity!")
            elif (Igoof == 6):
                print("Sorry to be the bearer of bad news, but your model \
                       has some problems.")
                print("You hit the center before the mass and/or luminosity \
                       were depleted.")
            else:
                print("CONGRATULATIONS, I THINK YOU FOUND IT!")
                

        Rcrat = r[istop]/Rs
        if (Rcrat < -9.999):
            Rcrat = -9.999
        Mcrat = M_r[istop]/Ms
        if (Mcrat < -9.999):
            Mcrat = -9.999
        Lcrat = L_r[istop]/Ls
        if (Lcrat < -9.999):
            Lcrat = -9.999

        print("The surface conditions are (Msolar,Mcrat,Rsolar,Rcrat,Lsolar,Lcrat,Te): ")
        print(self.Msolar,Mcrat,Rsolar,Rcrat,self.Lsolar, Lcrat,self.Te)
        print("The central conditions are (rhocor,X,Tcore,Y,Pcore,Z,epscor,dlPdlT[istop]): ")
        print(rhocor,self.X,Tcore,self.Y,Pcore,self.Z,epscor,dlPdlT[istop])

        fileout = open('starmodspy.out','w')
        results = csv.writer(fileout,delimiter=' ')
        results.writerow(["r","Qm","L_r","M_r","TT","PP","rho","kap","eps","clim","rcf","dlPdlT"])

        for ic in range(istop):
            i = istop - ic
#            print(i,ic,istop)
            Qm = 1.0-M_r[i]/Ms
            if (dlPdlT[i] < self.gamrat):
                rcf = "c"
            else:
                rcf = "c"
            if (np.abs(dlPdlT[i])>dlPlim):
                dlPdlT[i] = dlPlim*np.sign(dlPdlT[i])
                clim = "*"
            else:
                clim = " "
            results.writerow([r[i],Qm,L_r[i],M_r[i],T[i],P[i],rho[i],kappa[i],\
                              epsilon[i],clim,rcf,dlPdlT[i]])
                

        fileout.close()
        
        
#        print(r[200:202],M_r[200:202],np.array(M_r[200:202]))                
        self.plot(r,M_r,rho,T)

#        print(r,M_r,L_r,T,P,rho,kappa,epsilon)        



    def STARTMDL(self,deltar,mu,Rs,r_i,M_ri,L_ri,tog_bf,irc):
        r = r_i + deltar
        M_rip1 = M_ri
        L_rip1 = L_ri

        if (irc == 0):
            T_ip1 = self.G*M_rip1*mu*self.m_H/(4.25*self.k_B)*(1.0/r - 1.0/Rs)
            A_bf = 4.34e25*self.Z*(1.0 + self.X)/tog_bf
            A_ff = 3.68e22*self.g_ff*(1.0 - self.Z)*(1.0 + self.X)
            Afac = A_bf + A_ff
            P_ip1 = np.sqrt((1.0/4.25)*(16.0/3.0*np.pi*self.a*self.c)
                            *(self.G*M_rip1/L_rip1)
                            *(self.k_B/(Afac*mu*self.m_H)))*T_ip1**4.25
        else:
            T_ip1 = self.G*M_rip1*mu*self.m_H/self.k_B*(1.0/r - 1.0/Rs)/self.gamrat
            P_ip1 = kPad*T_ip1**self.gamrat            

#        print(irc,tog_bf,self.g_ff,Afac,A_bf,A_ff,T_ip1,P_ip1,self.G,M_rip1,L_rip1,mu,self.m_H,self.k_B,r,Rs,r_i,deltar)    
        sol = [r,P_ip1,M_rip1,L_rip1,T_ip1]
        return sol

    def EOS(self,mu,P,T,izone):
        ierr = 0
        if (T <= 0.0 or P <= 0.0):
            print("Error in EOS T or P: ",izone, T, P)

        Prad = (self.a*T**4.)/3.
        Pgas = P - Prad
        rho  = (mu*self.m_H/self.k_B)*(Pgas/T)
#        print(izone,P,T,Prad,Pgas,rho,mu,self.m_H,self.k_B,self.a)

        if (rho < 0.0):
            ierr = 1
            print("Error in EOS rho: ",izone, T, P, Prad, Pgas, rho)
        
        tog_bf = 2.82*(rho*(1.0 + self.X))**0.2
        k_bf = 4.34e25/tog_bf*self.Z*(1.0 + self.X)*rho/T**3.5
        k_ff = 3.68e22*self.g_ff*(1.0 - self.Z)*(1.0 + self.X)*rho/T**3.5
        k_e = 0.2*(1.0 + self.X)
        kappa = k_bf + k_ff + k_e

        T6 = T*1.0e-6
        oneo3 = 1./3.
        twoo3 = 2./3.
        fx = 0.133*self.X*np.sqrt((3.0 + self.X)*rho)/T6**1.5
        fpp = 1.0 + fx*self.X
        psipp = 1.0 + 1.412e8*(1.0/self.X - 1.0)*np.exp(-49.98*T6**(-oneo3))
        Cpp = 1.0 + 0.0123*T6**oneo3 + 0.0109*T6**twoo3 + 0.000938*T6
        epspp = 2.38e6*rho*self.X*self.X*fpp*psipp*Cpp*T6**(-twoo3)*\
                np.exp(-33.80*T6**(-oneo3))
        CCNO = 1.0 + 0.0027*T6**oneo3 - 0.00778*T6**twoo3 - 0.000149*T6
        epsCNO = 8.67e27*rho*self.X*self.XCNO*CCNO*T6**(-twoo3)*\
                 np.exp(-152.28*T6**(-oneo3))
        epsilon = epspp + epsCNO
        
        sol = [rho,kappa,epsilon,tog_bf,ierr]
        return sol

    def dMdr(self,r,rho):
        dMdr = 4.0*np.pi*rho*r**2
        return dMdr

    def dPdr(self,r, M_r,rho):
        dPdr = -self.G*rho*M_r/r**2
        return dPdr

    def dLdr(self,r,rho,epsilon):
        dLdr = 4.0*np.pi*rho*epsilon*r**2
        return dLdr

    def dTdr(self,r,M_r,L_r,T,rho,kappa,mu,irc):
        if (irc == 0):
            dTdr = -(3.0/(16.0*np.pi*self.a*self.c))*kappa*rho/T**3*L_r/r**2
        else:
            dTdr = -1.0/self.gamrat*self.G*M_r/r**2*mu*self.m_H/self.k_B
        return dTdr

    def RUNGE(self,f_im1,dfdr,r_im1,deltar,irc,mu,izone):
        dr12 = deltar/2.0
        dr16 = deltar/6.0
        r12 = r_im1 + dr12
        r_i = r_im1 + deltar

        f_temp = []
        for i in range(4):
            f_temp.append(f_im1[i]+dr12*dfdr[i])
        fndqsol = self.FUNDEQ(r12,f_temp,irc,mu,izone)
        df1  = fndqsol[0]
        ierr = fndqsol[1]
        if (ierr != 0):
            pass

        f_temp = []
        for i in range(4):
            f_temp.append(f_im1[i]+dr12*df1[i])
        fndqsol = self.FUNDEQ(r12,f_temp,irc,mu,izone)
        df2 = fndqsol[0]
        ierr = fndqsol[1]
        if (ierr != 0):
            pass            

        f_temp = []
        for i in range(4):
            f_temp.append(f_im1[i]+deltar*df2[i])
        fndqsol = self.FUNDEQ(r12,f_temp,irc,mu,izone)
        df3 = fndqsol[0]
        ierr = fndqsol[1]
        if (ierr != 0):
            pass             

        f_i = []
        for i in range(4):
            f_i.append(f_im1[i]+dr16*(dfdr[i]+2.0*df1[i]+2.0*df2[i]+df3[i])) 
        rng = [f_i,ierr]
        return rng

    def FUNDEQ(self,r,f,irc,mu,izone):
        P = f[0]
        M_r = f[1]
        L_r = f[2]
        T = f[3]

        eossol = self.EOS(mu,P,T,izone)

        rho = eossol[0]
        kappa = eossol[1]
        epsilon = eossol[2]
        ierr = eossol[4]
        
        dfdr = []
        dfdr.append(self.dPdr(r,M_r,rho))
        dfdr.append(self.dMdr(r,rho))
        dfdr.append(self.dLdr(r,rho,epsilon))
        dfdr.append(self.dTdr(r,M_r,L_r,T,rho,kappa,mu,irc))

        fndq = [dfdr,ierr]
        return fndq

    def rr2mm(self,position,rr,mr):
        rm = np.interp(position,rr,mr/self.Msun)
        print(rr[0:2])
#        print(rr[990],mr[990]/self.Msun)
        return np.round(rm,3)
    

    def plot(self,rr,mr,dd,tt):

#        prf = pd.read_csv('starmodspy.out',delimiter=' ',skiprows=1,names=['r','Qm','L_r','M_r','TT','P','rho','kap','eps','clim','rcf','dlPdlT'])

        prf = pd.read_csv('starmodspy.out',delimiter=' ')
        
        rr   = prf.r
        dd   = prf.rho
        tt   = prf.TT
        mr   = prf.M_r

        print(dd)
        
        ii = len(rr)
        
        fig, ax1 = plt.subplots(figsize=(7,6))
        ax1.plot(rr,dd,color='b',label=r'$\rho$')
        ax1.set_xlabel(r'r (cm)')
        ax1.set_ylabel(r'$\rho$ (g cm$^{-3})$')
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.plot(rr, tt,color='r',label=r'$T$')
        ax2.set_ylabel(r'T (K)')
        ax2.legend(loc=1,prop={'size':18})

        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        newxlabel=[rr[int(ii-1)],rr[int(ii/2.)],rr[int(ii/4)],rr[1]]
        ax3.set_xticks(newxlabel)
        ax3.set_xticklabels(self.rr2mm(newxlabel,rr,np.array(mr)))
        ax3.set_xlabel('enclosed mass (msol)')

        fig.tight_layout()        

        plt.show()
