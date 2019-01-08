"""

SynthesisModelSBS.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Jan  6 17:10:00 EST 2019

Description: 

"""

import sys
import numpy as np
from .Source import Source
import matplotlib.pyplot as pl
from ..util.ReadData import read_lit
from scipy.integrate import quad, cumtrapz
from ..util.Stats import bin_c2e, bin_e2c
from ..util.ParameterFile import ParameterFile
from ..physics.Constants import h_p, c, erg_per_ev, ev_per_hz, \
    s_per_yr, s_per_myr, Lsun, Tsun, g_per_msun, k_B
    

def _Planck(E, T):
    """ Returns specific intensity of blackbody at temperature T [K]."""

    nu = E * erg_per_ev / h_p
    return 2.0 * h_p * nu**3 / c**2 / (np.exp(h_p * nu / k_B / T) - 1.0)


class SynthesisModelSBS(Source):
    """
    Make this look like a SynthesisModel class. Could output to disk
    to save time...
    """
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
        self.fcore = 6e-3 * 0.74
        
        self.log10Mmin = self.pf['source_imf_bins'].min()
        self.log10Mmax = self.pf['source_imf_bins'].max()
        self.dlog10M = np.diff(self.pf['source_imf_bins'])[0]
        self.Mmin = 10**self.log10Mmin
        self.Mmax = 10**self.log10Mmax
        self.dM = np.diff(self.pf['source_imf_bins'])[0]
        self.aging = self.pf['source_stellar_aging']
    @property
    def tracks(self):
        if not hasattr(self, '_tracks'):
            if self.pf['source_tracks'] == 'parsec':
                mod = read_lit('parsec')
                self._tracks = mod._load(self.pf['source_tracks_fn'], 
                    self.masses)
            elif self.pf['source_tracks'] is not None:
                raise NotImplemented('help')
            else:
                self._tracks = None
        
        return self._tracks
        
    @property
    def _tab_life(self):
        if not hasattr(self, '_tab_life_'):
            assert self.tracks is not None
            
            self._tab_life_ = np.zeros_like(self.masses)
            for i, mass in enumerate(self.masses):
                ages = self.tracks['Age'][i]
                alive = np.logical_and(np.isfinite(ages), ages > 0)
                self._tab_life_[i] = ages[min(np.argwhere(~alive))-1]
            
        return self._tab_life_
        
    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            # Overkill for continuum, but still degraded 10x rel. to BPASS
            self._wavelengths = np.arange(30., 30010., 10.)
        return self._wavelengths
        
    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
        return self._energies
    
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = self.energies / h_p
        return self._frequencies
    
    @property
    def times(self):
        if not hasattr(self, '_times'):
            self._times = 10**np.arange(0., 4.1, 0.1)
        return self._times
    
    @property
    def masses(self):
        if not hasattr(self, '_masses'):
            self._masses = 10**self.pf['source_imf_bins']
        return self._masses
    
    def load(self):
        raise NotImplemented('help')
        
    def Spectrum(self, E, T):
        """ Returns specific intensity of blackbody at temperature T [K]."""
    
        nu = E * erg_per_ev / h_p
        return 2.0 * h_p * nu**3 / c**2 / (np.exp(h_p * nu / k_B / T) - 1.0)
    
    def _lum(self, M, t=None):
        """
        Luminosity of stars as function of their mass and age.
        
        Parameters
        ----------
        M : int, float, np.ndarray
            Mass [Msun]
        t : int, float, np.ndarray
            Time [Myr]
        """
        
        # Use tracks?
        if self.tracks is not None:
            if not self.aging:
                # Just use MS luminosity
                logL = np.interp(M, self.masses, self.tracks['logL'][:,0])
                return Lsun * 10**logL
            else:
                raise NotImplemented('help')
            #iM = np.argmin(np.abs(M - self.masses))
            #logL = np.interp(np.log10(t), self.tracks['Age'])
            
        ##    
        # Toy model
        ##
        if M < 0.43:
            return 0.23 * Lsun * M**2.3
        elif M < 2.:
            return Lsun * M**4
        elif M < 20.:
            return 1.4 * Lsun * M**3.5
        else:
            return 32e3 * Lsun * M
        
    def lum(self, M):
        if not hasattr(self, '_lum_func'):
            self._lum_func = np.vectorize(self._lum)
        return self._lum_func(M)
        
    def age(self, M):
        if self.tracks is not None:
            return np.interp(M, self.masses, self._tab_life)
        # If 'tracks' is not None, must tabulate this.
        
        return self.fcore * M * g_per_msun * c**2 / self.lum(M) / s_per_myr
 
    def temp(self, M):
        if self.tracks is not None:
            if not self.aging:
                # Just use MS luminosity
                logT = np.interp(M, self.masses, self.tracks['logTe'][:,0])
                return 10**logT
            else:
                raise NotImplemented('help')
                
        return Tsun * (M**2.5)**0.25
        
    @property
    def dldn(self):
        if not hasattr(self, '_dwdn'):
            l_edges = bin_c2e(self.wavelengths)
            e_edges = h_p * c / (l_edges / 1e8) / erg_per_ev
            n_edges = e_edges * erg_per_ev / h_p
            
            self._dldn = np.abs(np.diff(l_edges) / np.diff(n_edges))
            #self._dedn = np.diff(e_edges * erg_per_ev) / np.diff(n_edges)
            #self._dndl = np.diff(n_edges) / np.diff(l_edges)

        return self._dldn

    @property    
    def tab_Ls(self):
        """
        Tabulate spectra of stars.
        
        Units: erg/s/A
        """
        if not hasattr(self, '_tab_Ls'):
            
            l_edges = bin_c2e(self.wavelengths)
            e_edges = h_p * c / (l_edges / 1e8) / erg_per_ev
            n_edges = e_edges * erg_per_ev / h_p
            
            dedn = np.diff(e_edges * erg_per_ev) / np.diff(n_edges)
            dndl = np.diff(n_edges) / np.diff(l_edges)
            
            if self.tracks is not None and self.aging:
                self._tab_Ls = np.zeros((self.masses.size, 
                    self.wavelengths.size, self.times.size))
            else:
                self._tab_Ls = np.zeros((self.masses.size, 
                    self.wavelengths.size))
            
                
            for i, mass in enumerate(self.masses):
                
                if self.tracks is not None and self.aging:
                    pass
                else:    
                
                    T = self.temp(mass)
                    L = self.lum(mass)
                    
                    tot = quad(lambda EE: _Planck(EE, T), 0., np.inf)[0]
                    spec = self.Spectrum(self.energies, T) / erg_per_ev / tot
                    
                    # Damp UV emission from cooler stars?
                    corr = np.ones_like(spec)
                    
                    #if T < 1e4:
                    #    corr[self.wavelengths < 2e3] = 0
                    
                    spec *= corr
                    
                    self._tab_Ls[i,:] = L * spec * dedn * np.abs(dndl) 
            
        return self._tab_Ls    
            
    @property
    def data(self):
        """
        This is where we'll put the population-averaged spectra, i.e.,
        L as a function of wavelength and time.
        
        Units: erg/s/Ang/Msun
        
        """         
        if not hasattr(self, '_data'):
            ages = self.age(self.masses)
            self._data = np.zeros((self.wavelengths.size, self.times.size))
            for i, t in enumerate(self.times):
                
                alive = np.array(ages > t, dtype=int)
                
                #if self.tracks is not None:
                #    corr = 
                #    corr = np.array(ages > t, dtype=int)
                #else:
                #    corr = np.ones_like(ages)
                    
                # Recall that 'tab_L_ms' is 2-D, (mass, wavelength) if
                # not using stellar tracks, 3-D otherwise (mass, wavelength, time)
                L_per_dM = self.tab_imf[:,None] * alive[:,None] \
                         * self.tab_Ls

                self._data[:,i] = np.sum(L_per_dM * self.dM, axis=0) / 1e6

        return self._data

    def ngtm(self, m):
        return 1. - 10**np.interp(np.log10(m), np.log10(self.masses), np.log10(self.tab_imf_cdf))
        
    def mgtm(self, m):
        cdf_by_m = cumtrapz(self.tab_imf * self.masses**2, x=np.log(self.masses), initial=0.) \
            / np.trapz(self.tab_imf * self.masses**2, x=np.log(self.masses))
        
        return 1. - np.interp(m, self.masses, cdf_by_m)

    @property
    def tab_imf(self):
        """
        Normalized such that Mtot = 10^6 Msun.
        """
        if not hasattr(self, '_tab_imf'):
            if self.pf['source_imf'] in ['salpeter', 2.35]:
                # N = int_M1^M2 xi(M) dM where xi(M) = xi_0 * M**-2.35
                # 1e6 = int_M1^M2 M * xi_0 * M**-2.35 dM
                a = -2.35
                norm = (self.Mmax**(a + 2.) - self.Mmin**(a + 2.)) / (a + 2.)
                xi_0 = 1e6 / norm
                self._tab_imf = xi_0 * self.masses**a
            elif self.pf['source_imf'] == 'kroupa':
                m1 = 0.08; m2 = 0.5
                a0 = -0.3; a1 = -1.3; a2 = -2.3
                                                
                # Integrating to 10^6 Msun, hence two extra powers of M.                
                norm = ((m1**(a0 + 2.) - self.Mmin**(a0 + 2.)) / (a0 + 2.)) \
                     + (m1**a1 / m1**a2) * ((m2**(a1 + 2.) - m1**(a1 + 2.)) / (a1 + 2.)) \
                     + (m1**a1 / m1**a2) * (m2**a1 / m2**a2) * ((self.Mmax**(a2 + 2.) - m2**(a2 + 2.)) / (a2 + 2.))
                     
                _m0 = self.masses[self.masses < m1]
                _m1 = self.masses[np.logical_and(self.masses >= m1, self.masses < m2)]
                _m2 = self.masses[self.masses >= m2]     
                     
                n0 = self._n0 = 1e6 / norm
                n1 = self._n1 = n0 * m1**a1 / m1**a2
                n2 = self._n2 = n1 * m2**a1 / m2**a2
                
                i0 = n0 * _m0**a0
                i1 = n1 * _m1**a1
                i2 = n2 * _m2**a2
                
                self._tab_imf = np.concatenate((i0, i1, i2))
                
            elif self.pf['source_imf'] == 'chabrier':
                raise NotImplemented('help')
                xi = lambda M: 0.158 * (1. / np.log(10.) / M)
            else:
                raise NotImplemented('help')
            
        return self._tab_imf
    
    @property    
    def tab_imf_cdf(self):
        """
        CDF for IMF. By number, not mass!
        """
        if not hasattr(self, '_tab_imf_cdf'):
            # Doing this precisely is really important, so be careful!
            if self.pf['source_imf'] in ['salpeter', 2.35]:
                norm = (self.Mmin**-1.35 - self.Mmax**-1.35) / 1.35
                self._tab_imf_cdf = 1. - (self.masses**-1.35 - self.Mmax**-1.35) \
                    / 1.35 / norm
            elif self.pf['source_imf'] in ['kroupa']:
                
                # Poke imf to get coefficients
                poke = self.tab_imf
                
                m1 = 0.08; m2 = 0.5
                a0 = -0.3; a1 = -1.3; a2 = -2.3
                
                _m0 = self.masses[self.masses < m1]
                _m1 = self.masses[np.logical_and(self.masses >= m1, self.masses < m2)]
                _m2 = self.masses[self.masses >= m2]
                
                # Integrate up stars over all ranges.
                norm = self._n0 * ((m1**(a0 + 1.) - self.Mmin**(a0 + 1.)) / (a0 + 1.)) \
                     + self._n1 * ((m2**(a1 + 1.) - m1**(a1 + 1.)) / (a1 + 1.)) \
                     + self._n2 * ((self.Mmax**(a2 + 1.) - m2**(a2 + 1.)) / (a2 + 1.))
                
                # Stitch together CDF in different mass ranges.     
                _tot0 = ((_m0**(a0 + 1.) - self.Mmin**(a0 + 1.)) / (a0 + 1.))
                if _tot0.size == 0:
                    start = 0.0
                else:
                    start = _tot0[-1] 
                _tot1 = start + ((_m1**(a1 + 1.) - m1**(a1 + 1.)) / (a1 + 1.)) 
                _tot2 = _tot1[-1] + ((_m2**(a2 + 1.) - m2**(a2 + 1.)) / (a2 + 1.)) 
                
                self._tab_imf_cdf = np.concatenate((_tot0, _tot1, _tot2)) / _tot2[-1]
                
            else:    
                self._tab_imf_cdf = cumtrapz(self.tab_imf, x=self.masses, initial=0.) \
                    / np.trapz(self.tab_imf * self.masses, x=np.log(self.masses))

        return self._tab_imf_cdf
    
    @property
    def nsn_per_m(self):
        if not hasattr(self, '_nsn_per_m'):
            self._nsn_per_m = self.ngtm(8.) / self.mgtm(8.)
        return self._nsn_per_m
    
    def draw_stars(self, N):
        return np.interp(np.random.rand(N), self.tab_imf_cdf, self.masses)
            
            
            
