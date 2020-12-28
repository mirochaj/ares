"""
GalaxyHOD.py

Author: Emma Klemets
Affiliation: McGill University
Created on: June 3, 2020

Description: LF and SMF model (based on Moster2010), as well as main sequence SFR, SSFR and SFRD models (based on Speagle2014)

"""

from .Halo import HaloPopulation
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..util.ParameterFile import get_pq_pars
from ..util.MagnitudeSystem import MagnitudeSystem
from ..analysis.BlobFactory import BlobFactory
from ..physics.Constants import s_per_gyr
from ..physics.Cosmology import Cosmology

import numpy as np
from scipy.interpolate import interp1d

class GalaxyHOD(HaloPopulation, BlobFactory):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        HaloPopulation.__init__(self, **kwargs)

        
    def LuminosityFunction(self, z, x, text=False):
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
            f = interp1d(MUV, LF, kind='cubic', fill_value=-np.inf, bounds_error=False)    
            try:
                NumDensity = f(mags)
            except:
                # print("Error, magnitude(s) out of interpolation bounds")
                NumDensity = -np.inf * np.ones(len(mags))

        return NumDensity


    def Gen_LuminosityFunction(self, z, x, Lambda):
        """
        Reconstructed luminosity function for a given wavelength.
        **Only for Star-forming populations currently

        Population must be set with pars:
            pop_sed = 'eldridge2009' 
            pop_tsf = 12 -  population age [Myr]
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        x : float
            Absolute (AB) magnitudes.
        Lambda : float
            Wavelength in Ångströms
        
        Returns
        -------
        Number density.
        
        """

        if type(x) not in [list, np.ndarray]:
            mags = [x]
        else:
            mags = x

        Hm = self.halos.tab_M

        Lum = self.src.L_per_sfr(Lambda) * 10**self.SFR(z, Hm, True, log10=False) #[erg/s/Hz]
        
        k = np.argmin(np.abs(z - self.halos.tab_z))
        dndM = self.halos.tab_dndm[k, :][:-1]

        MagSys = MagnitudeSystem()
        MUV = MagSys.L_to_MAB(L=Lum, z=z)

        diff = []
        for i in range(len(MUV)-1):
            diff.append( (MUV[i+1] - MUV[i])/(Hm[i+1] - Hm[i]) )

        dLdM = np.abs(diff)
      
        LF = dndM/dLdM

        #check if requested magnitudes are in MUV, else interpolate LF function
        result =  all(elem in MUV for elem in mags)

        if result:
            #slice list to get the values requested
            findMags = np.array([elem in mags for elem in MUV])
            NumDensity = LF[findMags]
        else:
            f = interp1d(MUV[:-1], LF, kind='cubic', fill_value=-np.inf, bounds_error=False)    
            try:
                NumDensity = f(mags)
            except:
                NumDensity = -np.inf * np.ones(len(mags))

        return NumDensity


    def _dlogm_dM(self, N, M_1, beta, gamma):
        #derivative of log10( m ) wrt M for SMF
        
        dydx = -1* ((gamma-1)*(self.halos.tab_M/M_1)**(gamma+beta) - beta - 1) / (np.log(10)*self.halos.tab_M*((self.halos.tab_M/M_1)**(gamma+beta) + 1))

        return dydx


    def SMHM(self, z, log_HM, **kwargs):
        """
        Wrapper for getting stellar mass from a halo mass using the SMHM ratio. 
        """
        if log_HM == 0:
            haloMass = self.halos.tab_M
        elif type(log_HM) not in [list, np.ndarray]:
            haloMass = [10**log_HM]
        else:
            haloMass = [10**i for i in log_HM]
        

        N, M_1, beta, gamma = self._SMF_PQ()
        SM = self._SM_fromHM(z, haloMass, N, M_1, beta, gamma)

        return SM


    def HM_fromSM(self, z, log_SM, **kwargs):
        """
        For getting halo mass from a stellar mass using the SMHM ratio. 
        """

        haloMass = self.halos.tab_M

        N, M_1, beta, gamma = self._SMF_PQ()

        ratio = 2*N(z=z) / ( (haloMass/M_1(z=z))**(-beta(z=z)) + (haloMass/M_1(z=z))**(gamma(z=z)) )

        #just inverse the relation and interpolate, instead of trying to invert equ 2.
        f = interp1d(ratio*haloMass, haloMass, fill_value=-np.inf, bounds_error=False)

        log_HM = np.log10( f(10**log_SM))

        return log_HM


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
        #Gets the Parameterized Quantities for the SMF double power law
        #default values can be found in emma.py

        parsB = get_pq_pars(self.pf['pop_smhm_beta'], self.pf)
        parsN = get_pq_pars(self.pf['pop_smhm_n'], self.pf)
        parsG = get_pq_pars(self.pf['pop_smhm_gamma'], self.pf)
        parsM = get_pq_pars(self.pf['pop_smhm_m'], self.pf)

        N = ParameterizedQuantity(**parsN) #N_0 * (z + 1)**nu #PL
        M_1 = ParameterizedQuantity(**parsM) #10**(logM_0) * (z+1)**mu #different from Moster2010 paper
        beta = ParameterizedQuantity(**parsB) #beta_1*z+beta_0 #linear
        gamma = ParameterizedQuantity(**parsG) #gamma_0*(z + 1)**gamma_1 #PL

        return N, M_1, beta, gamma


    def _SF_fraction_PQ(self, sf_type, **kwargs):
        #Gets the Parameterized Quantities for the star-forming fraction tanh equation

        #default values can be found in emma.py

        parsA = get_pq_pars(self.pf['pop_sf_A'], self.pf)
        parsB = get_pq_pars(self.pf['pop_sf_B'], self.pf)

        parsC = get_pq_pars(self.pf['pop_sf_C'], self.pf)
        parsD = get_pq_pars(self.pf['pop_sf_D'], self.pf)

        A = ParameterizedQuantity(**parsA) 
        B = ParameterizedQuantity(**parsB)
        C = ParameterizedQuantity(**parsC)
        D = ParameterizedQuantity(**parsD)

        sf_fract = lambda z, Sh: (np.tanh(A(z=z)*(np.log10(Sh) + B(z=z))) + D(z=z))/C(z=z)

        SM = np.logspace(8, 12)
        test = sf_fract(z=1, Sh=SM)

        if sf_type == 'smf_tot':
            fract = lambda z, Sh: 1.0*Sh/Sh #the fraction is just 1, but it's still an array of len(Mh)

        elif any(i > 1 or i < 0 for i in test):
            # print("Fraction is unreasonable")
            fract = lambda z, Sh: -np.inf * Sh/Sh

        elif sf_type == 'smf_q':
            fract = lambda z, Sh: 1-sf_fract(z=z, Sh=Sh) # (1-sf_fract)

        else:
            fract = sf_fract

        return fract

   
    def StellarMassFunction(self, z, logbins, sf_type='smf_tot', text=False, **kwargs):
        """
        Stellar Mass Function from a double power law, following Moster2010
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        logbins : float
            log10 of Stellar mass bins. per stellar mass
        sf_type: string
            Specifies which galaxy population to use: total ='smf_tot' (default), 
            star-forming ='smf_sf', quiescent ='smf_q'
        
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
        sf_fract = self._SF_fraction_PQ(sf_type=sf_type)

        k = np.argmin(np.abs(z - self.halos.tab_z))

        StellarMass = self._SM_fromHM(z, haloMass, N, M_1, beta, gamma)
        SMF = hmf[k, :] * sf_fract(z=z, Sh=StellarMass) / self._dlogm_dM(N(z=z), M_1(z=z), beta(z=z), gamma(z=z)) #dn/dM / d(log10(m))/dM

        if np.isinf(StellarMass).all() or np.count_nonzero(StellarMass) < len(bins) or np.isinf(SMF).all():
            #something is wrong with the parameters and _SM_fromHM or _SF_fraction_PQ returned +/- infs,
            #or if there are less non-zero SM than SM values requested from bins

            if text:
                print("SM is inf or too many zeros!")
            phi = -np.inf * np.ones(len(bins))

        if np.array([i < 1e-1 for i in StellarMass]).all():
            if text:
                print("SM range is way too small!")
            phi = -np.inf * np.ones(len(bins))

        else:

            if len(StellarMass) != len(set(StellarMass)):
                #removes duplicate 0s from list
                if text:
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
                #values that are out of the range will return as -inf
                f = interp1d(np.log10(StellarMass), np.log10(SMF), kind='linear', fill_value=-np.inf, bounds_error=False)

                try:
                    phi = 10**(f(np.log10(bins)))

                except:
                    #catch if SM is completely out of the range
                    if text:
                        print("Error, bins out of interpolation bounds")
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

        #not returning error right now
        return SFRD[0]
    

    def SFR(self, z, logmass, haloMass=False, log10=True):   
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


        if log10:
            mass = [10**i for i in logmass]
        else:
            mass = logmass

        if haloMass:
            #convert from halo mass to stellar mass
            N, M_1, beta, gamma = self._SMF_PQ()

            Ms = self._SM_fromHM(z, mass, N, M_1, beta, gamma)
        else:
            Ms = mass

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
      
