"""

NebularEmission.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sun 21 Jul 2019 14:31:38 AEST

Description: 

"""

import numpy as np
from .Hydrogen import Hydrogen
from ..util import ParameterFile
from .Constants import h_p, c, k_B, erg_per_ev, E_LyA, E_LL, Ryd, ev_per_hz, \
    nu_alpha, m_p

class NebularEmission(object):
    def __init__(self, cosm=None, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.cosm = cosm
        
    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            raise AttributeError('Must set `wavelengths` by hand.')
        return self._wavelengths    
            
    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        
    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
        return self._energies    
        
    @property
    def Emin(self):
        return np.min(self.energies)
    
    @property
    def Emax(self):
        return np.max(self.energies)    
        
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies    
        
    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            self._dwdn = self.wavelengths**2 / (c * 1e8)
        return self._dwdn        
        
    @property
    def dE(self):
        if not hasattr(self, '_dE'):
            tmp = np.abs(np.diff(self.energies))
            self._dE = np.concatenate((tmp, [tmp[-1]]))
        return self._dE    
        
    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr    
        
    @property    
    def _gamma_fb(self):
        if not hasattr(self, '_gamma_fb_'):
            _gaunt_fb = 1.05
            _sum = np.zeros_like(self.frequencies)
            for n in xrange(2, 100, 1):
                _xn = Ryd / k_B / self.pf['pop_nebula_Tgas'] / n ** 2
                ok = (Ryd / h_p / n**2) < self.frequencies
                _sum[ok==1] += _xn * (np.exp(_xn) / n) * _gaunt_fb
            self._gamma_fb_ = 5.44e-39 * _sum
            
        return self._gamma_fb_
    
    @property
    def _gamma_ff(self):
        return 5.44e-39 * 1.1
        
    def _FreeBound(self, spec):
        e_fb = np.exp(-h_p * self.frequencies / k_B / self.pf['pop_nebula_Tgas'])
        return e_fb
        
    def _FreeFree(self, spec):
        e_ff = np.exp(-h_p * self.frequencies / k_B / self.pf['pop_nebula_Tgas'])
        return e_ff
        
    @property
    def _norm_free(self):
        if not hasattr(self, '_norm_free_'):
            integ = np.exp(-h_p * self.frequencies / k_B / self.pf['pop_nebula_Tgas'])
            self._norm_free_ = 1. / np.trapz(integ[-1::-1] * self.frequencies[-1::-1],
                x=np.log(self.frequencies[-1::-1]))
        return self._norm_free_        
        
    def _TwoPhoton(self, spec):
        x = self.energies / E_LyA

        P = np.zeros_like(self.energies)
        # Fernandez & Komatsu 2006
        P[x<1.] = 1.307 \
                - 2.627 * (x[x<1.] - 0.5)**2 \
                + 2.563 * (x[x<1.] - 0.5)**4 \
                - 51.69 * (x[x<1.] - 0.5)**6

        return 2. * self.energies * P / E_LyA / nu_alpha
        
    def f_rep(self, spec, channel='ff', net=False):
        """
        Fraction of photons reprocessed into different channels.
        """

        erg_per_phot = self.energies * erg_per_ev
        dE = self.dE * erg_per_ev

        if channel == 'ff':
            _ff = self._FreeFree(spec)
            frep = 4. * np.pi * self._gamma_ff * _ff / 2.06e-11
        elif channel == 'fb':
            _fb = self._FreeBound(spec)
            frep = 4. * np.pi * self._gamma_fb * _fb / 2.06e-11
        elif channel == 'tp':
            _tp = self._TwoPhoton(spec)
            frep = 2. * self.energies * erg_per_ev * _tp / nu_alpha
        else:
            raise NotImplemented("Do not recognize channel `{}`".format(channel))
    
        if net:
            return np.trapz(frep[-1::-1] * nu[-1::-1], x=np.log(nu[-1::-1]))
        else:
            return frep
            
    @property
    def is_ionizing(self):
        return self.energies >= E_LL
    
    def L_ion(self, spec):
        ion = self.energies >= E_LL
        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)

        return np.trapz(spec[ok==1][-1::-1] * self.energies[ok==1][-1::-1], 
            x=np.log(self.energies[ok==1][-1::-1]))

    def Continuum(self, spec, include_ff=True, include_fb=True,
        include_tp=True):
        """
        Add together nebular continuum contributions, i.e., free-free, 
        free-bound, and two-photon.

        Parameters
        ----------

        """
        
        fesc = self.pf['pop_fesc']
        Tgas = self.pf['pop_nebula_Tgas']
        flya = 2. / 3.
        erg_per_phot = self.energies * erg_per_ev
        
        ok = np.logical_and(self.is_ionizing, spec > 0)
        
        # This is in erg/s
        Lion = self.L_ion(spec)
        
        #               
        frep_ff = self.f_rep(spec, Tgas, 'ff')
        frep_fb = self.f_rep(spec, Tgas, 'fb')
        frep_tp = self.f_rep(spec, Tgas, 'tp')
        
        #_ff = self._FreeFree(spec)
        #_fb = self._FreeBound(spec)
        #_tp = self._TwoPhoton(spec)
        
        # Amount of UV luminosity absorbed in ISM
        Labs = Lion * (1. - fesc)
        
        # Normalize free-free and free-bound to total ionizing luminosity 
        # multiplied by their respective reprocessing factors. This is 
        # essentially just to get the result in the right units.
        norm_ff = 1. / np.trapz(frep_ff[-1::-1] * h_p * self.frequencies[-1::-1]**2, 
            x=np.log(self.frequencies[-1::-1])) / Labs
        norm_fb = 1. / np.trapz(frep_fb[-1::-1] * h_p * self.frequencies[-1::-1]**2, 
            x=np.log(self.frequencies[-1::-1])) / Labs
        norm_tp = 1. / np.trapz(frep_tp[-1::-1] * h_p * self.frequencies[-1::-1]**2, 
            x=np.log(self.frequencies[-1::-1])) / Labs
            
        tot = np.zeros_like(self.wavelengths)
        if include_ff:
            tot += frep_ff * norm_ff * Labs#(_ff * norm_ff)
        if include_fb:              
            tot += frep_fb * Labs#(_fb * norm_fb)
        if include_tp:
            tot += frep_tp * Labs#(_tp * norm_tp)
        
        return tot
                
    def LymanSeries(self, waves, spec):
        return self.HydrogenLines(waves, spec, ninto=1)
        
    def BalmerSeries(self, waves, spec):
        return self.HydrogenLines(waves, spec, ninto=2)
        
    def HydrogenLines(self, waves, spec, ninto=1):
        neb = np.zeros_like(waves)
        nrg = h_p * c / (waves * 1e-8) / erg_per_ev
        freq = nrg * erg_per_ev / h_p
        fesc = self.pf['pop_fesc']
        _Tg = self.pf['pop_nebula_Tgas']
                
        ion = nrg >= E_LL
        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)
        
        # This will be in [erg/s]
        Lion = np.trapz(spec[ok==1][-1::-1] * freq[ok==1][-1::-1], 
            x=np.log(freq[ok==1][-1::-1]))
        
        Lout = Lion * (1. - self.pf['pop_fesc'])
        sigm = nu_alpha * np.sqrt(k_B * _Tg / m_p / c**2) * h_p
        
        fout = np.zeros_like(waves)
        for n in range(ninto+1, 15):
            En = self.hydr.BohrModel(ninto=ninto, nfrom=n)
            
            prof = np.exp(-0.5 * (nrg - E_LyA)**2 / 2. / sigm**2) \
                 / np.sqrt(2. * np.pi) * erg_per_ev * ev_per_hz / sigm 
            
            # See if the line is resolved
            if not np.all(prof == 0):
                fout += Lout * frec * prof
            else:
                loc = np.argmin(np.abs(nrg - En))
                # Need to correct for size of bin
                corr = freq[loc-1] - freq[loc]
                fout[loc] += Lout * 2. / 3. / corr
            
                        
        return fout
        
    def LinesByElement(self, element='helium'):
        pass
        
        
        
        
        
        