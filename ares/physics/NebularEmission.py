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
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr    

    def _TwoPhoton(self, waves, spec):
        
        E = h_p * c / (waves * 1e-8) / erg_per_ev
        _sum = np.zeros_like(E)
        x = E/E_LyA # Note: x = E/E_LyA

        temp = np.zeros_like(E)
        temp[x<1.] = 1.307 \
                   - 2.627 * (x[x<1.] - 0.5)**2 \
                   + 2.563 * (x[x<1.] - 0.5)**4 \
                   - 51.69 * (x[x<1.] - 0.5)**6

        return E * temp / E_LyA

    def _FreeBound(self, waves, spec):
        E = h_p * c / (waves * 1e-8) / erg_per_ev
        nu = E * erg_per_ev / h_p
        _Tg = self.pf['pop_nebula_Tgas']
        _gaunt_ff = 1.1
        _gaunt_fb = 1.05
        _sum = np.zeros_like(E)
        for n in xrange(2, 100, 1):
            _xn = Ryd / k_B / _Tg / n ** 2
            ok = (Ryd / h_p / n**2) < nu
            _sum[ok==1] += _xn * np.exp(_xn) / n * _gaunt_fb
        gamma_c = 5.44e-39 * _sum
        #print '_sum:', _sum   # Explains why ff/fb parallel at 2nd plateau

        temp = gamma_c * np.exp(-h_p * nu / k_B / _Tg) * 4 * np.pi /2e-11

        return temp

    # ===== free-free ===== # 0.1-10.2 eV, reprocessed so 1-fesc = L
    def _FreeFree(self, waves, spec):
        """
        Free free
        """
        
        E = h_p * c / (waves * 1e-8) / erg_per_ev
        
        nu = E * erg_per_ev / h_p
        _Tg = self.pf['pop_nebula_Tgas']
        _gaunt_ff = 1.1
        _gaunt_fb = 1.05
        _sum = np.zeros_like(E)
        for n in xrange(2, 100, 1):
            _xn = Ryd / k_B / _Tg / n ** 2
            ok = (Ryd / h_p / n**2) < nu
            _sum[ok==1] += _xn * np.exp(_xn) / n * _gaunt_fb
                
        gamma_c = 5.44e-39 * _gaunt_ff

        temp = gamma_c * np.exp(-h_p * nu / k_B /_Tg) * 4. * np.pi / 2e-11

        return temp
        
    def Continuum(self, waves, spec):
        """
        Add together nebular continuum contributions, i.e., free-free, 
        free-bound, and two-photon.
        
        Paramete
        """
        
        neb = np.zeros_like(waves)
        nrg = h_p * c / (waves * 1e-8) / erg_per_ev
        fesc = self.pf['pop_fesc']
        
        ion = nrg >= E_LL
        gt0 = spec > 0
        ok = np.logical_and(ion, gt0)
        
        Lion = np.trapz(spec[ok==1][-1::-1] * nrg[ok==1][-1::-1], 
            x=np.log(nrg[ok==1][-1::-1]))
                
        _ff = self._FreeFree(waves, spec)
        norm_ff = 1. / (np.trapz(_ff[-1::-1] * nrg[-1::-1], 
            x=np.log(nrg[-1::-1])) / (1. - fesc) / Lion)
        
        _fb = self._FreeBound(waves, spec)
        norm_fb = 1. / (np.trapz(_fb[-1::-1] * nrg[-1::-1], 
            x=np.log(nrg[-1::-1])) / (1. - fesc) / Lion)
        
        _tp = self._TwoPhoton(waves, spec)
        norm_tp = 1. / (np.trapz(_tp[-1::-1] * nrg[-1::-1], 
            x=np.log(nrg[-1::-1])) / (1. - fesc) / Lion)    
            
        
        return _ff * norm_ff + _fb * norm_fb + _tp * norm_tp
        
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
        
        
        
        
        
        