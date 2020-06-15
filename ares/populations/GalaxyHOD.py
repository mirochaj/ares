
#the start of my ares class
from .Halo import HaloPopulation
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..util.ParameterFile import get_pq_pars
import numpy as np
from scipy.interpolate import interp1d

#Is there a built in thing in ARES for this?
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)


class GalaxyHOD(HaloPopulation):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        HaloPopulation.__init__(self, **kwargs)
        
    def LuminosityFunction(self, z, mags, text=True):
        """
        Reconstructed luminosity function from a simple model of L = c*HaloMadd
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        mags : bool
            If True, x-values will be in absolute (AB) magnitudes
        
        Returns
        -------
        Number density.
        
        """

        #get halo mass function and array of halo masses
        hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        #might be overkill here
        # pars = {}
        # pars['pq_func'] = 'linear' # double power-law with evolution in norm
        # pars['pq_func_var'] = 'z'
        # pars['pq_func_par0'] = 3e-4
        # pars['pq_func_par1'] = 0
        # pars['pq_func_par2'] = 0

        # c = ParameterizedQuantity(**pars) #really just a constant

        #LF loglinear models
        c = 3e-4
        k = np.argmin(np.abs(z - self.halos.tab_z))
        
        LF = (np.log(10)*haloMass)/2.5 * hmf[k, :]
        MUV = -2.5*np.log10(c*haloMass)

        #check if requested magnitudes are in MUV, else interpolate LF function
        result =  all(elem in MUV for elem in mags)

        if result:
            #slice list to get the values requested
            findMags = np.array([elem in mags for elem in MUV])
            NumDensity = LF[findMags]
        else:
            if text:
                print("Interpolating")
            f = interp1d(MUV, LF, kind='cubic')    

            NumDensity = f(mags)

        return NumDensity

    def _dlogm_dM(self, N, M_1, beta, gamma):
        #derivative of log10( m ) wrt M for SMF
        
        dydx = -1* ((gamma-1)*(self.halos.tab_M/M_1)**(gamma+beta) - beta - 1) / (np.log(10)*self.halos.tab_M*((self.halos.tab_M/M_1)**(gamma+beta) + 1))

        return dydx


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

        #could have these as defaults for variables passed?

        # Npq = 0
        # Nparam = 0
        # pqs = []
        # for kwarg in kwargs:

        #     if isinstance(kwargs[kwarg], basestring):
        #         if kwargs[kwarg][0:2] == 'pq':
        #             Npq += 1
        #             pqs.append(kwarg)
        #     elif (kwarg in parametric_options) and (kwargs[kwarg]) is not None:
        #         Nparam += 1

        #From Moster2010, table 7 - eventually user should be able to change these (also do fits so default ones are better)
        logM_0 = 11.88 #(0.01)
        mu = 0.019 #(0.002)
        N_0 = 0.0282 #(0.0003)
        nu = -0.72 #(0.06)
        gamma_0 = 0.556 #0.001
        gamma_1 = -0.26 #(0.05)
        beta_0 = 1.06 #(0.06)
        beta_1 = 0.17 #(0.12)


        parsB = get_pq_pars(self.pf['pop_smhm_beta'], self.pf)

        parsN = get_pq_pars(self.pf['pop_smhm_n'], self.pf)

        parsG = get_pq_pars(self.pf['pop_smhm_gamma'], self.pf)

        parsM = get_pq_pars(self.pf['pop_smhm_m'], self.pf)

        N = ParameterizedQuantity(**parsN) #N_0 * (z + 1)**nu #PL
        M_1 = ParameterizedQuantity(**parsM) #10**(logM_0*(z+1)**mu)
        beta = ParameterizedQuantity(**parsB) #beta_1*z+beta_0 #linear
        gamma = ParameterizedQuantity(**parsG) #gamma_0*(z + 1)**gamma_1 #PL

        return N, M_1, beta, gamma

    
    def StellarMassFunction(self, z, bins, text=True, **kwargs):
        """
        Stellar Mass Function from a double power law, following Moster2010
        
        Parameters
        ----------
        z : int, float
            Redshift. Currently does not interpolate between values in halos.tab_z if necessary.
        bins : bool
            Stellar mass bins. per stellar mass
        
        Returns
        -------
        Phi : float (array)
            Number density of galaxies [cMpc^-3 dex^-1]
        """

        #get halo mass function and array of halo masses
        hmf = self.halos.tab_dndm
        haloMass = self.halos.tab_M

        N, M_1, beta, gamma = self._SMF_PQ()

        k = np.argmin(np.abs(z - self.halos.tab_z))

        SMF = hmf[k, :] / self._dlogm_dM(N(z=z), M_1(z=z), beta(z=z), gamma(z=z)) #dn/dM / d(log10(m))/dM
        StellarMass = self._SM_fromHM(z, haloMass, N, M_1, beta, gamma)

        #check if requested mass bins are in StellarMass, else interpolate SMF function
        result =  all(elem in StellarMass for elem in bins)

        if result:
            #slice list to get the values requested
            findMass = np.array([elem in bins for elem in StellarMass])
            phi = SMF[findMass]           
        else:
            #interpolate
            if text:
                print("Interpolating")
            f = interp1d(StellarMass, SMF, kind='cubic')
            #ADD error catch if SM is out of the range
            phi = f(bins)

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
            numberD = self.StellarMassFunction(zi, SM_bins, False)

            SFR = 10**self.SFR(zi, SM_bins)/SM_bins
            error = 0.2 * SFR * np.log(10)

            dbin = []
            for i in range(0, len(SM_bins) - 1):
                dbin.append(SM_bins[i+1]-SM_bins[i])

            SFRD_val = np.sum( numberD[:-1] * SFR[:-1] * dbin )
            SFRD_err = np.sqrt(np.sum( numberD[:-1] * dbin * error[:-1])**2)
            
            SFRD.append([SFRD_val, SFRD_err])

        SFRD = np.transpose(SFRD) # [sfrd, err]

        return SFRD
    

    def SFR(self, z, mass, haloMass=False):   
        """
        Main sequence stellar formation rate from Speagle2014
        
        Parameters
        ----------
        z : int, float
            Redshift.
        mass : float (array)
            if haloMass=False (default) is the stellar masses [stellar mass]
            else halo masses [stellar mass]
        
        Returns
        -------
        logSFR : float (array)
            log10 of MS SFR [yr^-1]
        """

        # cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)

        if haloMass:
            #convert from halo mass to stellar mass
            N, M_1, beta, gamma = self._SMF_PQ()
            Ms = self._SM_fromHM(z, mass, N, M_1, beta, gamma)
        else:
            Ms = mass

        # t: age of universe in Gyr
        t = cosmo.age(z).value

        if t < cosmo.age(6).value: # if t > z=6
            print("Warning, age out of well fitting zone of this model.")

        error = np.ones(len(Ms)) * 0.2 #[dex] the stated "true" scatter
        logSFR = (0.84-0.026*t)*np.log10(Ms) - (6.51-0.11*t) #Equ 28

        return logSFR

    #specific sfr
    def SSFR(self, z, mass, haloMass=False):
        """
        Specific stellar formation rate.
        
        Parameters
        ----------
        z : int, float
            Redshift.
        mass : float (array)
            if haloMass=False (default) is the stellar masses [stellar mass]
            else halo masses [stellar mass]
        
        Returns
        -------
        logSSFR : float (array)
            log10 of SSFR [yr^-1]
        """

        if haloMass:
            #convert from halo mass to stellar mass
            N, M_1, beta, gamma = self._SMF_PQ()
            Ms = self._SM_fromHM(z, mass, N, M_1, beta, gamma)
        else:
            Ms = mass

        logSSFR = self.SFR(z, Ms) - np.log10(Ms)

        return logSSFR
      