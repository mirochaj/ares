"""

BlackHolePopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 26 20:11:01 2012

Description: 

"""

import types
import numpy as np
from ..sources import BlackHole
from scipy.integrate import quad
from ..util import ParameterFile
from .Halo import HaloPopulation
from scipy.misc import derivative
from ..util.PrintInfo import print_pop
from ..util.NormalizeSED import norm_sed
from ..physics.Cosmology import Cosmology
from ..physics.Constants import rhodot_cgs
from ..physics.SecondaryElectrons import SecondaryElectrons

E_th = [13.6, 24.6, 54.4]

class BlackHolePopulation:
    def __init__(self, grid=None, init_rs=True, **kwargs):
        """
        Initialize black hole population.          
        """
                
        self.pf = ParameterFile(**kwargs)
        self.kwargs = kwargs
        
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
            
        self.switch = False
        if self.pf['source_sed'] == 'simpl':
            self.pf.update({'source_sed': 'mcd'})
            self.switch = True
            
        self.spectrum = BlackHole(**self.pf)
        
        self.esec = SecondaryElectrons(method=self.pf["secondary_ionization"])
                                        
        self.model = self.pf['model']

        self.zform = self.pf["formation_redshift"]
        self.zdead = self.pf["extinction_redshift"]
        self.zfl = self.pf['first_light_redshift']
        
        # Last redshift we'll consider (for anything)
        self.zoff = self.pf['zoff']
                      
        self.approx_src = self.pf['approx_lwb'] and self.pf['approx_xrb']

        # Re-normalize bolometric output
        self._init_rs()
            
        # Print info to screen
        if self.pf['verbose']:
            print_pop(self)    
            
        # Halo mass function stuff
        self._init_pop()
        
        # Pre-tabulate mass-density (if required)
        self._initialize_BHs()
    
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
    
    def _set_fcoll(self, Tmin, mu):
        self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
            self.halos.build_1d_splines(Tmin, mu)
    
    def _init_rs(self):
        sed = norm_sed(self, self.grid)
        
        if self.switch:
            self.pf.update({'source_sed': 'simpl'})
        
        self.spectrum = BlackHole(grid=self.grid, 
            init_tabs=False, **self.pf)
        
        self.rs = self.spectrum#sed['rs']
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
                
    def _init_pop(self):
        # Halo stuff
        if self.pf['sfrd'] is not None:
            return
        
        if self.pf['fcoll'] is None and self.model != 3:
            self.halos = HaloPopulation(**self.pf)
            self._set_fcoll(self.pf['Tmin'], self.pf['mu'])
        else:
            self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
                self.pf['fcoll'], self.pf['dfcolldz'], self.pf['d2fcolldz2']
        
        # Compute second derivatives                
        self._d2fcolldz2 = lambda z: derivative(self.dfcolldz, z)
                    
        # Model 3 requires this, hence if statement above
        if self.model == 3:
            self.dz = np.diff(self.halos.z)[0]
        
    def _initialize_BHs(self):
        """
        Construct BH mass density as a function of redshift, or at least some
        quantities needed to compute it later.
        """    
          
        # X-ray binaries  
        if self.model < 0:
            pass
            
        # 'Burst'
        elif self.model == 0:
            self.rho_bh_zform = self.pf['rhoi']() / rho_cgs

        # BH mass accretion rate density = const. fraction of collapsing gas density    
        elif self.model == 1:
            pass
        
        # BH mass density = const. fraction of collapsed mass density
        elif self.model == 2:
            pass    
                                
        # BH mass density tied to formation rate density of progenitors 
        # and Eddington-limited growth    
        elif self.model == 3:
            L = int(1 + (self.zform - self.zdead) / self.dz)   # Number of formation redshifts
            M = int(1 + (self.zform - self.zoff) / self.dz)    # Number of total redshift bins
            self.zforming = np.linspace(self.zdead, self.zform, L)
            self.zall = np.linspace(self.zoff, self.zform, M)
            self.rho_bh_z = np.zeros(M)
                                                        
            # Set up dt array - must account for all formation redshifts
            # This is the time between formation redshift z_l and redshift z_m
            self.dt_lm = np.zeros([len(self.zforming), len(self.zall)])
            for l, zl in enumerate(self.zforming):
                for m, zm in enumerate(self.zall):
                    if zl <= zm:
                        continue
                        
                    self.dt_lm[l][m] = self.cosm.LookbackTime(zm, zl)
                    
            # Formation rate density
            self.bh_frd = np.zeros_like(self.zforming)
            for l, zl in enumerate(self.zforming):
                self.bh_frd[l] = self.FormationRateDensity(zl) * self.cosm.dtdz(zl)
                            
            # Mass density when we go from forming new black holes to purely
            # exponential growth of the population                
            self.rho_bh_ztrans = np.trapz(self.bh_frd * \
                np.exp((1. - self.pf['eta']) * self.dt_lm[:,M - L] * \
                self.pf['fedd'] / self.pf['eta'] / t_edd), self.zforming)
                                
            # Calculate black hole mass density as a function of redshift
            for m, zm in enumerate(self.zall):
                if zm >= self.zdead:
                    self.rho_bh_z[m] = np.trapz(self.bh_frd[m - (M - L):] * \
                        np.exp((1. - self.pf['eta']) * self.dt_lm[m - (M - L):,m] * \
                        self.pf['fedd'] / self.pf['eta'] / t_edd), self.zall[m:])
                else:
                    self.rho_bh_z[m] = self.rho_bh_ztrans * \
                        np.exp((1. - self.pf['eta']) * self.dt_lm[0, m] * self.pf['fedd'] / self.pf['eta'] / t_edd)        
                                        
    def XrayLuminosityDensity(self, z):
        """
        Return total luminosity density of this population at redshift z in units 
        of erg / s / cm^3 (proper cm^3).
        
        This is called "XrayLuminosityDensity" to be consistent with
        StellarPopulation class, even though it is the bolometric luminosity
        density.
        """
        
        if not (self.zdead <= z <= self.zform) or (z > self.zfl):
            return 0.0
            
        if self.pf['emissivity'] is not None:
            return self.pf['emissivity'](z)    
        
        if self.model < 0:
            if self.pf['xi_XR'] is not None:
                return self.cX * self.pf['xi_XR'] * self.SFRD(z) \
                    / self.pf['fstar']
                    
            return self.cX * self.pf['fX'] * self.SFRD(z)
            
        elif self.model == 1:
            return self.pf['fedd'] * self.pf['eta'] * self.AccretionRateDensity(z) * c**2 / (1. - self.pf['eta'])
        else:
            return self.MassDensity(z) * self.pf['fedd'] * c**2 / t_edd
            
    def AccretionRateDensity(self, z):
        """
        Return comoving mass accretion rate density in g / s / cm^3 at redshift z.
        """
        
        if self.model == 1:
            return self.cosm.rho_b_z0 * self.pf['fbh'] * self.dfcolldz(z) / self.cosm.dtdz(z)
        
        return self.MassDensity(z) * self.pf['fedd'] * (1. - self.pf['eta']) / self.pf['eta'] / t_edd
        
    def TotalGrowthRateDensity(self, z):
        """
        Return 'anti-accretion rate density' required in order to preserve 
        self-consistency.  Assumes constant (wrt z) accretion physics parameters.
        """
        
        drhoBHdz = (self.MassDensity(z-0.05) - self.MassDensity(z+0.05)) / 0.1
        return drhoBHdz / self.cosm.dtdz(z)
        
        #return self.cosm.rho_b_z0 * self.pf['fbh'] \
        #    * (self.pf['eta'] * t_edd / (1. - self.pf['eta']) / self.pf['fedd']) \
        #    * self.cosm.hubble_0**2 * self.cosm.omega_m_0 * (1. + z)**5 \
        #    * (2.5 * self.dfcolldz(z) / (1. + z) + self.d2fcolldz2(z))
                    
    def AccretionlessGrowthRateDensity(self, z):
        """
        Return comoving (total) growth rate density in g / s / cm^3 at redshift z.
        """ 
        
        return self.TotalGrowthRateDensity(z) - self.AccretionRateDensity(z)
             
    def MassDensity(self, z):
        """
        Compute comoving BH mass density in g / cm^3 at redshift z.
        """
        
        if (z > self.zform):
            return 0.0
        
        # Black-hole evolution treated in a more complex fashion
        if self.model == 0:
            return self.rho_bh_zform * \
                np.exp((1. - self.pf['eta']) * self.cosm.LookbackTime(z, self.zform) * \
                self.pf['fedd'] / self.pf['eta'] / t_edd)  
        
        # BH mass accretion rate density = const. fraction of collapsing gas density    
        elif self.model == 1:
            return self.pf['eta'] * t_edd * self.cosm.rho_b_z0 * self.pf['fbh'] * \
                self.dfcolldz(z) / self.cosm.dtdz(z) / (1. - self.pf['eta'])
        
        # BH mass density = const. fraction of collapsed mass density
        elif self.model == 2:
            return self.cosm.rho_m_z0 * self.fcoll(z) * self.pf['fbh']
        
        elif self.model == 3: # Use spline instead - pickle it to use later?
            return np.interp(z, self.zall, self.rho_bh_z, left=0.0, right=0.0)
        
        else:
            return 0.0
                       
    def FormationRateDensity(self, z):
        """
        Compute rate of BH formation in units of g / cm^3 / s.
        This will not be called for FormationMethod = (0, 1, 2).
        This quantity is comoving.
        """       
        
        if not (self.zdead <= z <= self.zform):
            return 0.0
            
        if self.model == 0:
            if z == self.zform:
                return self.rho_bh_zform / self.cosm.dtdz(z)
            else:
                return 0.0
        elif self.model == 1: # A
            pass
        elif self.model == 2: # B
            pass    
        elif self.model == 3: # C
            return self.pf['fbh'] * self.SFRD(z)      

    def SFRD(self, z):
        """
        Return comoving star-formation rate density at redshift z in g / s / cm^3.
        """
        
        if not (self.zdead <= z <= self.zform):
            return 0.0
            
        # SFRD approximated by some analytic function    
        if self.pf['sfrd'] is not None:
            return self.pf['sfrd'](z) / rhodot_cgs
                    
        return self.pf['fstar'] * self.cosm.rho_b_z0 * self.dfcolldz(z) / self.cosm.dtdz(z)
            
    def MassFunction(self, M, z, Mi=1e2):
        """
        Compute black hole mass function.
        
        Parameters
        ----------
        M : float
            Mass (in solar units) to consider
        z : float
            Redshift at which to compute the BHMF.
        Mi : float
            Seed mass of BHs in solar units.
            
        Returns
        -------
        Differential density of black holes of mass M at redshift z in units
        of number per comoving Mpc**3.
        
        """
        if not (self.model == 3):
            raise ValueError('Can only derive BHMF for model 3.')

        zform = self.FormationRedshift(M, z, Mi=Mi)
        if zform > self.zform or np.isnan(zform):
            return 0.0
            
        return self.FormationRateDensity(zform) * self.cosm.dtdz(zform) \
            * cm_per_mpc**3 / M / g_per_msun
            
    def FormationRedshift(self, M, z, Mi=1e2):
        if not (self.model == 3):
            raise ValueError('Can only derive BHMF for model 3.')
            
        t = self.cosm.LookbackTime(z, 1e7)
        tform = t - np.log(M / Mi) * self.pf['eta'] / self.pf['fedd'] \
            * t_edd / (1. - self.pf['eta'])
            
        return self.cosm.TimeToRedshiftConverter(t, tform, z)        
            
    def BlackHoleMass(self, Mi, t, fedd=1, eta=0.1):
        """
        Compute black hole mass after t (seconds) have elapsed.  Relies on 
        initial mass self.M, and (constant) radiative efficiency self.epsilon.
        """        
        
        return Mi * np.exp(((1.0 - eta) / eta) * t * fedd / t_edd)         
            
    def _Emissivity(self, z, E):
        """
        Compute comoving volume emissivity of population at redshift z and energy E in units
        of erg s^-1 cm^-3 Hz**-1.
        """    
                                
        return self.XrayLuminosityDensity(z) * self.rs.Spectrum(E) * ev_per_hz 
            
    def _LymanWernerEmissivity(self, z, E):
        """
        Emissivity non-ionizing photons.
        """
        
        return self.XrayLuminosityDensity(z) * self.rs.Spectrum(E) * ev_per_hz 
    
    def _NumberEmissivity(self, z, E):
        return self._Emissivity(z, E) / E / erg_per_ev
        
    def _LymanWernerNumberEmissivity(self, z, E):
        return self._LymanWernerEmissivity(z, E) / E / erg_per_ev
        
    def Emissivity(self, z, E):
        """
        Return emissivity from whatever part of spectrum is most appropriate.
        """
        
        if E < E_LL:
            return self._LymanWernerEmissivity(z, E)
        else:
            return self._Emissivity(z, E)
    
    def NumberEmissivity(self, z, E):
        return self.Emissivity(z, E) / E / erg_per_ev

        
    