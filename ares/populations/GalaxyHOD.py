"""
GalaxyHOD.py

Author: Emma Klemets
Affiliation: McGill University
Created on: June 3, 2020

Description: LF and SMF model based off Moster2010, as well as main sequence SFR, SSFR and SFRD models (based on Speagle2014)

"""

from .Halo import HaloPopulation
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..util.ParameterFile import get_pq_pars
from ..analysis.BlobFactory import BlobFactory
from ..physics.Constants import s_per_gyr
from ..physics.Cosmology import Cosmology

import numpy as np
from scipy.interpolate import interp1d

class GalaxyHOD(HaloPopulation, BlobFactory):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        HaloPopulation.__init__(self, **kwargs)
        
    def LuminosityFunction(self, z, x, text=True):
        """
        Reconstructed luminosity function from a simple model of L = c*HaloMadd
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        x : float
            Absolute (AB) magnitudes.
        
        Returns
        -------
        Number density.
        
        """

        #catch if only one magnitude is passed
        if type(x) not in [list, np.ndarray]:
            mags = [x]
        else:
            mags = x

        #get halo mass function and array of halo masses
        hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        #default is really just a constant, c = 3e-4
        pars = get_pq_pars(self.pf['pop_lf'], self.pf)
        c = ParameterizedQuantity(**pars)

        #LF loglinear models
        k = np.argmin(np.abs(z - self.halos.tab_z))
        
        LF = (np.log(10)*haloMass)/2.5 * hmf[k, :]
        MUV = -2.5*np.log10(c(z=z)*haloMass)

        #check if requested magnitudes are in MUV, else interpolate LF function
        result =  all(elem in MUV for elem in mags)

        if result:
            #slice list to get the values requested
            findMags = np.array([elem in mags for elem in MUV])
            NumDensity = LF[findMags]
        else:
            # if text:
            #     print("Interpolating")
            f = interp1d(MUV, LF, kind='cubic')    
            try:
                NumDensity = f(mags)
            except:
                # print("Error, magnitude(s) out of interpolation bounds")
                NumDensity = -np.inf * np.ones(len(mags))

        return NumDensity

    def _dlogm_dM(self, N, M_1, beta, gamma):
        #derivative of log10( m ) wrt M for SMF
        
        dydx = -1* ((gamma-1)*(self.halos.tab_M/M_1)**(gamma+beta) - beta - 1) / (np.log(10)*self.halos.tab_M*((self.halos.tab_M/M_1)**(gamma+beta) + 1))

        return dydx

    def SMHM(self, z, **kwargs):
        # hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        N, M_1, beta, gamma = self._SMF_PQ()

        # k = np.argmin(np.abs(z - self.halos.tab_z))

        SM = self._SM_fromHM(z, haloMass, N, M_1, beta, gamma)

        return SM

    def _SM_fromHM(self, z, haloMass, N, M_1, beta, gamma):
        """
        Using the SMHM ratio, given a halo mass, returns the corresponding stellar mass

        Parameters
            ----------
            z : int, float
                Redshift.
            haloMass : float
                per stellar mass
            N, M_1, beta, gamma : Parameterized Quantities
                Dependant on z

        """

        mM_ratio = np.log10( 2*N(z=z) / ( (haloMass/M_1(z=z))**(-beta(z=z)) + (haloMass/M_1(z=z))**(gamma(z=z)) ) ) #equ 2

        StellarMass = 10**(mM_ratio + np.log10(haloMass))

        return StellarMass

    def _SMF_PQ(self, **kwargs):

        #could have these as defaults can be found in emma.py

        parsB = get_pq_pars(self.pf['pop_smhm_beta'], self.pf)
        parsN = get_pq_pars(self.pf['pop_smhm_n'], self.pf)
        parsG = get_pq_pars(self.pf['pop_smhm_gamma'], self.pf)
        parsM = get_pq_pars(self.pf['pop_smhm_m'], self.pf)

        N = ParameterizedQuantity(**parsN) #N_0 * (z + 1)**nu #PL
        M_1 = ParameterizedQuantity(**parsM) #10**(logM_0*(z+1)**mu)
        beta = ParameterizedQuantity(**parsB) #beta_1*z+beta_0 #linear
        gamma = ParameterizedQuantity(**parsG) #gamma_0*(z + 1)**gamma_1 #PL

        return N, M_1, beta, gamma

    
    def StellarMassFunction(self, z, logbins, text=True, **kwargs):
        """
        Stellar Mass Function from a double power law, following Moster2010
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        logbins : float
            log10 of Stellar mass bins. per stellar mass
        
        Returns
        -------
        Phi : float (array)
            Number density of galaxies [cMpc^-3 dex^-1]
        """

        #catch if only one magnitude is passed
        if type(logbins) not in [list, np.ndarray]:
            bins = [10**logbins]
        else:
            bins = [10**i for i in logbins]

        #get halo mass function and array of halo masses
        hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        N, M_1, beta, gamma = self._SMF_PQ()

        k = np.argmin(np.abs(z - self.halos.tab_z))

        SMF = hmf[k, :] / self._dlogm_dM(N(z=z), M_1(z=z), beta(z=z), gamma(z=z)) #dn/dM / d(log10(m))/dM
        StellarMass = self._SM_fromHM(z, haloMass, N, M_1, beta, gamma)

        # print("SMF first")
        # print(SMF)

       # print("SM now")
        # print(StellarMass)

        if StellarMass[-1] < 1e-10:
            print(StellarMass)

        """this guy is the problem
            maybe only catch if all the SM are very small (like above for printing)
            else, just cut out the values that are pretty small, and don't interp those ones,
            as my bins aren't going to be in that range anyways
        """

        if np.isinf(StellarMass).all() or np.count_nonzero(StellarMass) < len(bins):
            #something is wrong with the parameters and _SM_fromHM returned +/- infs
            #if there are less non-zero SM than SM values requested from bins
            print("SM is inf or too many zeros!")
            phi = -np.inf * np.ones(len(bins))

        if np.array([i < 1e-1 for i in StellarMass]).all():
            print("SM range is way too small!")
            phi = -np.inf * np.ones(len(bins))

        else:

            #removes duplicate 0s from list
            if len(StellarMass) != len(set(StellarMass)):
                print("removing some zeros")
                removeMask = [0 != i for i in StellarMass]
                
                StellarMass = StellarMass[removeMask]
                SMF = SMF[removeMask]

            #check if requested mass bins are in StellarMass, else interpolate SMF function
            result =  all(elem in StellarMass for elem in bins)

            if result:
                #slice list to get the values requested
                findMass = np.array([elem in bins for elem in StellarMass])
                phi = SMF[findMass]           
            else:
                #interpolate
                # if text:
                #     print("Interpolating")

                # f = interp1d(StellarMass, SMF, kind='cubic')
                f = interp1d(np.log10(StellarMass), np.log10(SMF), kind='linear')

                #ADD error catch if SM is out of the range
                try:
                    # phi = f(bins)
                    phi = 10**(f(np.log10(bins)))

                except:
                    print("Error, bin(s) out of interpolation bounds")
                    phi = -np.inf * np.ones(len(bins))

        return phi    
        
    def SFRD(self, z):
        """
        Stellar formation rate density.
        
        Parameters
        ----------
        z : int, float (array)
            Redshift.
        
        Returns
        -------
        SFRD : float (array)
            [M_o/yr/Mpc^3]
        """

        #population comes from halo and SMF
        hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        N, M_1, beta, gamma = self._SMF_PQ()

        #Check if z is only a single value - will only return one value
        if type(z) not in [list, np.ndarray]:
            z = [z]

        SFRD = []

        for zi in z:
            SM_bins = self._SM_fromHM(zi, haloMass, N, M_1, beta, gamma)

            #get number density
            numberD = self.StellarMassFunction(zi, np.log10(SM_bins), False)

            SFR = 10**self.SFR(zi, np.log10(SM_bins))/SM_bins
            error = 0.2 * SFR * np.log(10)

            dbin = []
            for i in range(0, len(SM_bins) - 1):
                dbin.append(SM_bins[i+1]-SM_bins[i])

            SFRD_val = np.sum( numberD[:-1] * SFR[:-1] * dbin )
            SFRD_err = np.sqrt(np.sum( numberD[:-1] * dbin * error[:-1])**2)
            
            SFRD.append([SFRD_val, SFRD_err])

        SFRD = np.transpose(SFRD) # [sfrd, err]


        return SFRD[0]
    

    def SFR(self, z, logmass, haloMass=False):   
        """
        Main sequence stellar formation rate from Speagle2014
        
        Parameters
        ----------
        z : int, float
            Redshift.
        mass : float (array)
            if haloMass=False (default) is the log10 stellar masses [stellar mass]
            else log10 halo masses [stellar mass]
        
        Returns
        -------
        logSFR : float (array)
            log10 of MS SFR [yr^-1]
        """

        if haloMass:
            #convert from halo mass to stellar mass
            N, M_1, beta, gamma = self._SMF_PQ()
            mass = [10**i for i in logmass]
            Ms = self._SM_fromHM(z, mass, N, M_1, beta, gamma)
        else:
            Ms = [10**i for i in logmass]

        cos = Cosmology()

        # t: age of universe in Gyr
        t = cos.t_of_z(z=z) / s_per_gyr

        if t < cos.t_of_z(z=6) / s_per_gyr: # if t > z=6
            print("Warning, age out of well fitting zone of this model.")

        error = np.ones(len(Ms)) * 0.2 #[dex] the stated "true" scatter

        pars1 = get_pq_pars(self.pf['pop_sfr_1'], self.pf)
        pars2 = get_pq_pars(self.pf['pop_sfr_2'], self.pf)

        func1 = ParameterizedQuantity(**pars1)
        func2 = ParameterizedQuantity(**pars2)

        logSFR = func1(t=t)*np.log10(Ms) - func2(t=t) #Equ 28


        # logSFR = (0.84-0.026*t)*np.log10(Ms) - (6.51-0.11*t) #Equ 28

        return logSFR

    #specific sfr
    def SSFR(self, z, logmass, haloMass=False):
        """
        Specific stellar formation rate.
        
        Parameters
        ----------
        z : int, float
            Redshift.
        mass : float (array)
            if haloMass=False (default) is the log10 stellar masses [stellar mass]
            else log10 halo masses [stellar mass]
        
        Returns
        -------
        logSSFR : float (array)
            log10 of SSFR [yr^-1]
        """

        if haloMass:
            #convert from halo mass to stellar mass
            N, M_1, beta, gamma = self._SMF_PQ()
            mass = [10**i for i in logmass]
            Ms = self._SM_fromHM(z, mass, N, M_1, beta, gamma)
        else:
            Ms = [10**i for i in logmass]

        logSSFR = self.SFR(z, np.log10(Ms)) - np.log10(Ms)

        return logSSFR
      
