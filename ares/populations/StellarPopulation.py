"""

StellarPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 26 20:11:05 2012

Description: 

"""

import types, sys
import numpy as np
from ..physics import Cosmology
from scipy.integrate import quad
from ..util import ParameterFile
from ..physics.Constants import *
from ..util.PrintInfo import print_pop
from ..util.NormalizeSED import norm_sed
from .HaloMassFunction import HaloDensity
from ..util.Warnings import negative_SFRD

class StellarPopulation:
    def __init__(self, grid=None, init_rs=True, **kwargs):
        """
        Initialize a stellar-population object.
        
        Parameters
        ----------
        grid : instance of static.Grid.Grid class
        init_rs : bool
            Initialize an sources.RadiationSource object?
        
        """
        self.pf = ParameterFile(**kwargs)
                
        self.grid = grid
           
        if grid is None:
            self.cosm = Cosmology(
                omega_m_0=self.pf['omega_m_0'], 
                omega_l_0=self.pf['omega_l_0'], 
                omega_b_0=self.pf['omega_b_0'],  
                hubble_0=self.pf['hubble_0'],  
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf['cmb_temp_0'], 
                approx_highz=self.pf['approx_highz'], 
                sigma_8=self.pf['sigma_8'], 
                primordial_index=self.pf['primordial_index'])
        else:
            self.cosm = grid.cosm
            
        self.model = -1   
            
        # Baryons per gram - use this for LyA emissivity (assumes mu=1.22)
        self.b_per_g = 1. / self.cosm.g_per_baryon

        self.burst = np.equal(*self.pf["formation_epoch"])
        self.zform = max(self.pf["formation_epoch"])
        self.zdead = min(self.pf["formation_epoch"])
        self.zfl = self.pf['first_light_redshift']
                
        self.approx_src = (self.pf['approx_lwb'] and self.pf['approx_xrb'])

        # Re-normalize bolometric output
        self._init_rs()

        # Print info to screen
        print_pop(self)
        
    @property
    def fcoll(self):
        if not hasattr(self, '_fcoll'):
            self._init_pop()
        
        return self._fcoll
    
    @property
    def dfcolldz(self):
        if not hasattr(self, '_dfcolldz'):
            self._init_pop()
    
        return self._dfcolldz
        
    @property
    def d2fcolldz2(self):
        if not hasattr(self, '_d2fcolldz2'):
            self._init_pop()
    
        return self._d2fcolldz2

    def _set_fcoll(self, Tmin, mu):
        self.Tmin = Tmin
        self.pf.update({'Tmin': Tmin})
        self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
            self.halos.build_1d_splines(Tmin, mu)

    def _init_pop(self):
        
        if self.pf['emissivity'] is not None:
            return
        if self.pf['sfrd'] is not None:
            return
        
        # Halo stuff            
        if self.pf['fcoll'] is None:
            self.halos = HaloDensity(**self.pf)
            self._set_fcoll(self.pf['Tmin'], self.pf['mu'])
        else:            
            self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
                self.pf['fcoll'], self.pf['dfcolldz'], self.pf['d2fcolldz2']

    def _init_rs(self):
        """
        Initialize RadiationSource (rt1d) instance - normalize LW, UV, and
        X-ray luminosities appropriately.
        """
        
        sed = norm_sed(self, self.grid)
        
        self.rs = sed['rs']
        self.cX = sed['cX']
        self.cLW = sed['cLW']
        self.cUV = sed['cUV']
        self.erg_per_LW = sed['erg_per_LW']
        self.erg_per_UV = sed['erg_per_UV']
        self.erg_per_X = sed['erg_per_X'] 
        self.Nlw = sed['Nlw']
        self.Nion = sed['Nion']
        self.Nx = sed['Nx']
        self.Elw = sed['Elw']
        self.Eion = sed['Eion']
        self.Ex = sed['Ex']
                
    def update(self, rs=True, pop=True, **kwargs):
        """
        Update contents of parameter file. 
        
        Initializing a StellarPopulation object can take a few seconds, 
        primarily because (sometimes) we have to read in a lookup table for
        the collapsed fraction and create splines. 
        
        If Tmin isn't changing but spectral parameters are, this routine
        saves us some time by only updating spectral parameters.
        
        Note that if the parameter 'sfrd' is being used, we'll update that too
        because it's cheap.
        
        """

        tmp = self.pf.copy()
        for key in kwargs:
            tmp.update({key:kwargs[key]})
        
        self.pf = tmp
        
        if pop:
            self._init_pop()
        if rs:
            self._init_rs()       
    
    def fcoll_dot(self, z):
        return self.dfcolldz(z) / self.cosm.dtdz(z)
    
    def rho_b_dot(self, z):   
        return self.cosm.rho_b_z0 * self.fcoll_dot(z)

    def SFRD(self, z):
        """
        Compute the comoving star formation rate density (SFRD).
        
        Given that we're in the StellarPopulation class, we are assuming
        that all emissivities are tied to the star formation history. The
        SFRD can be supplied explicitly as a function of redshift, or can 
        be computed via the "collapsed fraction" formalism. That is, compute
        the SFRD given a minimum virial temperature of star forming halos 
        (Tmin) and a star formation efficiency (fstar).
        
        If supplied as a function, the units should be Msun yr**-1 cMpc**-3.
        
        If fstar is None, will just return the rate of baryonic collapse.
        
        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        Co-moving star-formation rate density at redshift z in units of
        g s**-1 cm**-3.

        """
        
        if (z > self.zform) or (z > self.zfl):
            return 0.0
            
        # SFRD approximated by some analytic function    
        if self.pf['sfrd'] is not None:
            return self.pf['sfrd'](z) / rhodot_cgs
               
        if self.pf['fstar'] is None:
            return self.rho_b_dot(z)
                           
        # SFRD computed via fcoll parameterization
        sfrd = self.pf['fstar'] * self.rho_b_dot(z)
                                               
        if sfrd < 0:
            negative_SFRD(z, self.pf['Tmin'], self.pf['fstar'], 
                self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
            sys.exit(1)
                                   
        return sfrd                           
                                   
    def LymanWernerLuminosityDensity(self, z):
        """
        Compute comoving non-ionizing luminosity density.
        
        Calculate the total energy output between Lyman-alpha and the
        Lyman-limit (I know, not really the Lyman-Werner band, so sue me).
        
        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        Comoving luminosity density at redshift z in units of
        erg s**-1 (comoving cm)**-3.
        
        """
                   
        if self.pf['xi_LW'] is not None:
            return self.cLW * self.pf['xi_LW'] * self.SFRD(z) \
                / self.pf['Nlw']

        return self.cLW * self.SFRD(z)
    
    def LymanWernerPhotonLuminosityDensity(self, z):
        """
        Compute comoving non-ionizing photon luminosity density.
        
        Calculate the total number of photons emitted between Lyman-alpha and 
        the Lyman-limit (I know, not really the Lyman-Werner band, so sue me).
                
        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        Comoving luminosity density at redshift z in s**-1 (comoving cm)**-3.
        
        """

        return self.LymanWernerLuminosityDensity(z) / self.erg_per_LW

    def _LymanWernerEmissivity(self, z, E):
        """
        Emissivity of non-ionizing photons in energy units,
        erg s**-1 cm**-3 Hz**-1
        """

        return self.LymanWernerLuminosityDensity(z) * self.rs.Spectrum(E) \
            * ev_per_hz
            
    def _LymanWernerNumberEmissivity(self, z, E):
        """ s**-1 cm**-3 Hz**-1"""

        return self._LymanWernerEmissivity(z, E) / E / erg_per_ev
    
    #def IonizingLuminosityDensity(self, z):
    #    """
    #    Compute comoving ionizing luminosity density.
    #
    #    Parameters
    #    ----------
    #    z : float
    #        redshift
    #
    #    Returns
    #    -------
    #    Comoving luminosity density at redshift z in units of
    #    erg s**-1 (comoving cm)**-3.
    #
    #    """
    #
    #    return self.cUV * self.pf['fion'] * self.SFRD(z)
    
    def IonizingPhotonLuminosityDensity(self, z):
        """
        Compute comoving ionizing luminosity density (but not X-rays).
    
        Parameters
        ----------
        z : float
            redshift
    
        Returns
        -------
        Comoving luminosity density at redshift z in units of
        photons s**-1 (comoving cm)**-3.
    
        """
        
        if self.pf['xi_UV'] is not None:
            return self.b_per_g * self.pf['xi_UV'] * self.SFRD(z)
                
        return self.Nion * self.pf['fesc'] * self.b_per_g * self.SFRD(z)
    
    def XrayLuminosityDensity(self, z):
        """
        Compute (bolometric) comoving luminosity density.
        
        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        Luminosity density in units of erg s**-1 (comoving cm)**-3.
        
        """
        
        if self.pf['emissivity'] is not None:
            return self.pf['emissivity'](z)
        
        if self.pf['xi_XR'] is not None:
            return self.cX * self.pf['xi_XR'] * self.SFRD(z)

        return self.cX * self.pf['fX'] * self.SFRD(z)
        
    def _XrayEmissivity(self, z, E):
        """
        Compute comoving volume emissivity of population.
        
        Parameters
        ----------
        z : float
            redshift
        E : float
            emission energy (eV)
        
        Returns
        -------    
        Emissivity in units of erg s**-1 Hz**-1 cm**-3.
        """
                                
        return self.XrayLuminosityDensity(z) * self.rs.Spectrum(E) * ev_per_hz
    
    def _XrayNumberEmissivity(self, z, E):
        """
        Compute comoving volume (number) emissivity of population.
        
        Parameters
        ----------
        z : float
            redshift
        E : float
            emission energy (eV)
        
        Returns
        -------    
        Emissivity in units of photons s**-1 Hz**-1 cm**-3.
        
        See also
        --------
        XrayEmissivity
        """
        
        return self._XrayEmissivity(z, E) / E / erg_per_ev
        
    def XrayTotalNumberEmissivity(self, z):
        return quad(lambda EE: self._XrayNumberEmissivity(z, EE),
            self.rs.Emin, self.rs.Emax)[0] / ev_per_hz
        
    def Emissivity(self, z, E):
        """
        Return co-moving emissivity from whatever part of spectrum is 
        most appropriate in erg s**-1 cm**-3.
        """
        
        if E < E_LL:
            return self._LymanWernerEmissivity(z, E)
        else:
            return self._XrayEmissivity(z, E)
    
    def NumberEmissivity(self, z, E):
        """
        Return co-moving emissivity in photons / s / cm^3.
        """
        
        if E < E_LL:
            return self._LymanWernerNumberEmissivity(z, E)
        else:
            return self._XrayNumberEmissivity(z, E)
