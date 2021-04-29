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


class SynthesisModelSBS(Source): # pragma: no cover
    """
    Make this look like a SynthesisModel class. Could output to disk
    to save time...
    """
    def __init__(self, **kwargs):
        #self.pf = ParameterFile(**kwargs)
        Source.__init__(self, **kwargs)
        
        self.fcore = 6e-3 * 0.74

        self.aging = self.pf['source_stellar_aging']
        
    def __getattr__(self, name):
        if (name[0] == '_'):
            if name.startswith('_tab'):
                return self.__getattribute__(name)
                
            raise AttributeError('Couldn\'t find attribute: {!s}'.format(name))
                
        poke = self.Ms
        
        return self.__dict__[name]
        
    @property
    def tracks(self):
        if not hasattr(self, '_tracks'):
            if self.pf['source_tracks'] in ['parsec', 'eldridge2009']:
                mod = read_lit(self.pf['source_tracks'])
                self._tracks = mod._load_tracks(**self.pf)
            elif self.pf['source_tracks'] is not None:
                raise NotImplemented('help')
            else:
                self._tracks = None
        
        return self._tracks
        
    @property
    def tab_life(self):
        if not hasattr(self, '_tab_life_'):
            assert self.tracks is not None

            self._tab_life_ = np.zeros_like(self.Ms)
            for i, mass in enumerate(self.Ms):
                
                if self.pf['source_tracks'] == 'eldridge2009':
                    tracks = self.tracks[mass]
                    self._tab_life_[i] = max(tracks['age']) / 1e6
                else:   
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
    def Ms(self):
        if not hasattr(self, '_Ms'):
            if self.pf['source_tracks'] == 'eldridge2009':
                self._Ms = self.tracks['masses']
            else:
                self._Ms = 10**self.pf['source_imf_bins']
                
            self.log10Mmin = np.log10(self._Ms).min()        
            self.log10Mmax = np.log10(self._Ms).max()        
            self.dlog10M = np.diff(np.log10(self._Ms))[0]    
            self.Mmin = 10**self.log10Mmin                           
            self.Mmax = 10**self.log10Mmax
            
            # kludgey. Interpolate to uniform grid?
            self.dM = np.concatenate((np.diff(self._Ms), [0]))
                
        return self._Ms
        
    @property
    def Ms_e(self):
        """
        Bin edges.
        """
        
        if not hasattr(self, '_Ms_e'):
            dM = np.diff(self.Ms)
            
            if np.allclose(np.diff(dM), 0):
                self._Ms_e = bin_c2e(self.Ms)
            else:
                # Be more careful for non-uniform binning.    
                #assert self.pf['source_tracks'] == 'eldridge2009'
                raise NotImplemented('help')
                
                    
        return self._Ms_e        
            
    
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
                logL = np.interp(M, self.Ms, self.tracks['logL'][:,0])
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
        
    #def lum(self, M):
    #    if not hasattr(self, '_lum_func'):
    #        self._lum_func = np.vectorize(self._lum)
    #    return self._lum_func(M)
    #    
    #def age(self, M):
    #    if self.tracks is not None:
    #        return np.interp(M, self.masses, self._tab_life)
    #    # If 'tracks' is not None, must tabulate this.
    #    
    #    return self.fcore * M * g_per_msun * c**2 / self.lum(M) / s_per_myr
    #
    #def temp(self, M):
    #    if self.tracks is not None:
    #        if not self.aging:
    #            # Just use MS luminosity
    #            logT = np.interp(M, self.masses, self.tracks['logT'][:,0])
    #            return 10**logT
    #        else:
    #            raise NotImplemented('help')
    #            
    #    return Tsun * (M**2.5)**0.25
        
    #@property
    #def dldn(self):
    #    if not hasattr(self, '_dwdn'):
    #        l_edges = bin_c2e(self.wavelengths)
    #        e_edges = h_p * c / (l_edges / 1e8) / erg_per_ev
    #        n_edges = e_edges * erg_per_ev / h_p
    #        
    #        self._dldn = np.abs(np.diff(l_edges) / np.diff(n_edges))
    #        #self._dedn = np.diff(e_edges * erg_per_ev) / np.diff(n_edges)
    #        #self._dndl = np.diff(n_edges) / np.diff(l_edges)
    #
    #    return self._dldn
    #
    @property
    def tab_LUV(self):
        if not hasattr(self, '_tab_LUV'):
            Ls = self.tab_Ls
            Ly = self.wavelengths <= 912.
            Lall = Ls[:,Ly==1]
            
            # Still function of mass
            self._tab_LUV = np.trapz(Lall, x=self.wavelengths[Ly==1], axis=1)
            
        return self._tab_LUV
            
    @property    
    def tab_Ls(self):
        """
        Tabulated spectra of stars.
        
        Units: erg/s/A
        """
        if not hasattr(self, '_tab_Ls'):
            
            l_edges = bin_c2e(self.wavelengths)
            e_edges = h_p * c / (l_edges / 1e8) / erg_per_ev
            n_edges = e_edges * erg_per_ev / h_p
            
            dedn = np.diff(e_edges * erg_per_ev) / np.diff(n_edges)
            dndl = np.diff(n_edges) / np.diff(l_edges)
            
            if self.tracks is not None and self.aging:
                self._tab_Ls = np.zeros((self.Ms.size, 
                    self.wavelengths.size, self.times.size))
            else:
                self._tab_Ls = np.zeros((self.Ms.size, 
                    self.wavelengths.size))
                    
            if self.tracks is not None:

                if self.pf['source_tracks'] == 'eldridge2009':
                    if self.aging:
                        k = slice(0,None,1)
                    else:   
                        k = 0
                    
                    A = [self.tracks[m]['age'][k] \
                        for m in self.tracks['masses']]
                    T = [10**self.tracks[m]['logT'][k] \
                        for m in self.tracks['masses']]
                    L = [Lsun * 10**self.tracks[m]['logL'][k] \
                        for m in self.tracks['masses']]    
                else:    
                    T = 10**self.tracks['logTe'][:,0]
                    L = Lsun * 10**self.tracks['logL'][:,0]
            else:
                T = self.temp(self.Ms)
                L = self.lum(self.Ms)

            for i, mass in enumerate(self.Ms):

                if self.aging:    

                    Loft = np.interp(self.times, A[i] / 1e6, L[i], right=0.)
                    Toft = np.interp(self.times, A[i] / 1e6, T[i], right=0.)
                    
                    tot = quad(lambda EE: _Planck(EE, Toft[0]), 0., np.inf)[0]
                    spec = self.Spectrum(self.energies, Toft[0]) / erg_per_ev / tot
                    
                    self._tab_Ls[i] = Loft * spec[:,None] * dedn[:,None] \
                        * np.abs(dndl)[:,None]
                else:    
                    tot = quad(lambda EE: _Planck(EE, T[i]), 0., np.inf)[0]
                    spec = self.Spectrum(self.energies, T[i]) / erg_per_ev / tot
                    self._tab_Ls[i] = L[i] * spec * dedn * np.abs(dndl) 

        return self._tab_Ls
            
    @property
    def data(self):
        """
        This is where we'll put the population-averaged spectra, i.e.,
        L as a function of wavelength and time.
        
        Units: erg/s/Ang/Msun
        
        """         
        if not hasattr(self, '_data'):
            
            if self.pf['source_tracks'] == 'eldridge2009':
                ages = self.tab_life
            else:    
                ages = self.age(self.Ms)
            
            self._data = np.zeros((self.wavelengths.size, self.times.size))
            for i, t in enumerate(self.times):
                
                
                if self.aging:
                    # (mass, wavelength, time)
                    Ls = self.tab_Ls[:,:,i]
                else:
                    alive = np.array(ages > t, dtype=int)
                    Ls = self.tab_Ls * alive[:,None]
                
                #if self.tracks is not None:
                #    corr = 
                #    corr = np.array(ages > t, dtype=int)
                #else:
                #    corr = np.ones_like(ages)
                
                # Recall that 'tab_L_ms' is 2-D, (mass, wavelength) if
                # not using stellar tracks, 3-D otherwise (mass, wavelength, time)
                L_per_dM = self.tab_imf[:,None] * Ls

                self._data[:,i] = np.sum(L_per_dM * self.dM[:,None], axis=0) / 1e6

        return self._data

    def ngtm(self, m):
        return 1. - 10**np.interp(np.log10(m), np.log10(self.Ms), np.log10(self.tab_imf_cdf))
        
    def mgtm(self, m):
        cdf_by_m = cumtrapz(self.tab_imf * self.Ms**2, x=np.log(self.Ms), initial=0.) \
            / np.trapz(self.tab_imf * self.Ms**2, x=np.log(self.Ms))
        
        return 1. - np.interp(m, self.Ms, cdf_by_m)

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
                self._tab_imf = xi_0 * self.Ms**a
            elif self.pf['source_imf'] == 'kroupa':
                m1 = 0.08; m2 = 0.5
                a0 = -0.3; a1 = -1.3; a2 = -2.3
                                                
                # Integrating to 10^6 Msun, hence two extra powers of M.                
                norm = ((m1**(a0 + 2.) - self.Mmin**(a0 + 2.)) / (a0 + 2.)) \
                     + (m1**a1 / m1**a2) \
                     * ((m2**(a1 + 2.) - m1**(a1 + 2.)) / (a1 + 2.)) \
                     + (m1**a1 / m1**a2) * (m2**a1 / m2**a2) \
                     * ((self.Mmax**(a2 + 2.) - m2**(a2 + 2.)) / (a2 + 2.))
                     
                _m0 = self.Ms[self.Ms < m1]
                _m1 = self.Ms[np.logical_and(self.Ms >= m1, self.Ms < m2)]
                _m2 = self.Ms[self.Ms >= m2]     
                     
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
                self._tab_imf_cdf = 1. - (self.Ms**-1.35 - self.Mmax**-1.35) \
                    / 1.35 / norm
            elif self.pf['source_imf'] in ['kroupa']:
                
                # Poke imf to get coefficients
                poke = self.tab_imf
                
                m1 = 0.08; m2 = 0.5
                a0 = -0.3; a1 = -1.3; a2 = -2.3
                
                _m0 = self.Ms[self.Ms < m1]
                _m1 = self.Ms[np.logical_and(self.Ms >= m1, self.Ms < m2)]
                _m2 = self.Ms[self.Ms >= m2]
                
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
                self._tab_imf_cdf = cumtrapz(self.tab_imf, x=self.Ms, initial=0.) \
                    / np.trapz(self.tab_imf * self.Ms, x=np.log(self.Ms))

        return self._tab_imf_cdf
    
    @property
    def nsn_per_m(self):
        if not hasattr(self, '_nsn_per_m'):
            self._nsn_per_m = self.ngtm(8.) / self.mgtm(8.)
        return self._nsn_per_m
    
    def draw_stars(self, N):
        return np.interp(np.random.rand(N), self.tab_imf_cdf, self.Ms)
            
    #def tab_sn_dtd(self):
    #    """
    #    Delay time distribution.
    #    """
    #    if not hasattr(self, '_tab_sn_dtd'):
    #        self._tab_sn_dtd = np.zeros_like(self.times)
    #        
    #        self.tab_life self.tab_imf

    @property
    def max_sn_delay(self):
        if not hasattr(self, '_max_sn_delay'):
            self._max_sn_delay = float(self.tab_life[self.Ms == 8.])
        return self._max_sn_delay
    
    @property
    def min_sn_delay(self):
        if not hasattr(self, '_min_sn_delay'):
            self._min_sn_delay = float(self.tab_life[-1])
        return self._min_sn_delay    
            
    @property        
    def avg_sn_delay(self):
        if not hasattr(self, '_avg_sn_delay'):
            ok = self.Ms >= 8.
            top = np.trapz(self.tab_life[ok==1] * self.tab_imf[ok==1], 
                x=self.Ms[ok==1])
                
            bot = np.trapz(self.tab_imf[ok==1], x=self.Ms[ok==1])
            
            self._avg_sn_delay = top / bot
        
        return self._avg_sn_delay
    
    @property
    def tab_dtd_cdf(self):
        if not hasattr(self, '_tab_dtd_cdf'):
            ok = self.Ms >= 8.
            top = cumtrapz(self.tab_life[ok==1] * self.tab_imf[ok==1] \
                * self.Ms[ok==1], x=np.log(self.Ms[ok==1]), initial=0.0)
                
            bot = np.trapz(self.tab_life[ok==1] * self.tab_imf[ok==1] \
                * self.Ms[ok==1], x=np.log(self.Ms[ok==1]))
            
            self._tab_dtd_cdf = top / bot
            
        return self._tab_dtd_cdf
            
    def draw_delays(self, N):
        ok = self.Ms >= 8.
        return np.interp(np.random.rand(N), self.tab_dtd_cdf, 
            self.tab_life[ok==1])
        
    #@property        
    #def var_sn_delay(self):
    #    if not hasattr(self, '_var_sn_delay'):
    #        ok = self.Ms >= 8.
    #        top = np.trapz(self.tab_life[ok==1] * self.tab_imf[ok==1], 
    #            x=self.Ms[ok==1])
    #
    #        bot = np.trapz(self.tab_imf[ok==1], x=self.Ms[ok==1])
    #
    #        self._avg_sn_delay = top / bot
    #
    #    return self._avg_sn_delay    
            
