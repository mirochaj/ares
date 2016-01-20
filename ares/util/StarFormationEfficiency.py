"""

StarFormationEfficiency.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import numpy as np

z0 = 9. # arbitrary

class ParameterizedSFE(object):

    @property
    def Mfunc(self):
        return self.pf.pfs[self.pop_id]['sfe_Mfun']
    
    @property
    def zfunc(self):
        return self.pf.pfs[self.pop_id]['sfe_zfun']
    
        
    @property
    def fpeak(self):
        if not hasattr(self, '_fpeak'):
            self._fpeak = self.func('fpeak')
    
        return self._fpeak
        
    @property
    def Mpeak(self):
        if not hasattr(self, '_Mpeak'):
            self._Mpeak = self.func('Mpeak')
    
        return self._Mpeak  
    
    @property
    def sigma(self):
        if not hasattr(self, '_sigma'):
            self._sigma = self.func('sigma')
    
        return self._sigma     
    
    @property
    def Mlo_extrap(self):
        if not hasattr(self, '_Mlo_extrap'):
            self._Mlo_extrap = self.pf['sfe_Mfun_lo'] is not None
        return self._Mlo_extrap
    @property
    def Mhi_extrap(self):
        if not hasattr(self, '_Mhi_extrap'):
            self._Mhi_extrap = self.pf['sfe_Mfun_hi'] is not None
        return self._Mhi_extrap   
    
    def func(self, name):        
        if self.pf['sfe_%s' % name] == 'constant':
            func = lambda zz: self.pf['sfe_%s_par0' % name]
        elif self.pf['sfe_%s' % name] == 'linear_z':
            coeff1 = self.pf['sfe_%s_par0' % name]
            coeff2 = self.pf['sfe_%s_par1' % name]
            func = lambda zz: coeff1 + coeff2 * (1. + zz) / (1. + z0)
        elif self.pf['sfe_%s' % name] == 'linear_t':
            coeff = self.pf['sfe_%s_par0' % name]
            func = lambda zz: 10**(np.log10(coeff) - 1.5 * (1. + zz) / (1. + z0))
        elif self.pf['sfe_%s' % name] == 'pl':
            coeff1 = self.pf['sfe_%s_par0' % name]
            coeff2 = self.pf['sfe_%s_par1' % name]
            func = lambda zz: 10**(np.log10(coeff1) + coeff2 * (1. + zz) / (1. + z0))
        elif self.pf['sfe_%s' % name] == 'poly':
            coeff1 = self.pf['sfe_%s_par0' % name]
            coeff2 = self.pf['sfe_%s_par1' % name]
            coeff3 = self.pf['sfe_%s_par2' % name]
            func = lambda zz: 10**(np.log10(coeff1) + coeff2 * (1. + zz) / (1. + z0) \
                + coeff3 * ((1. + zz) / (1. + z0))**2)

        return func

    @property
    def _apply_extrap(self):
        if not hasattr(self, '_apply_extrap_'):
            self._apply_extrap_ = 1
        return self._apply_extrap_

    @_apply_extrap.setter
    def _apply_extrap(self, value):
        self._apply_extrap_ = value    
        
    def fstar(self, z, M):
        """
        Compute the star formation efficiency.
        """
        
        logM = np.log10(M)
        
        if self.Mfunc == 'lognormal':            
            f = self.fpeak(z) * np.exp(-(logM - np.log10(self.Mpeak(z)))**2 \
                / 2. / self.sigma(z)**2)
        elif self.Mfunc == 'poly':
            raise NotImplemented('sorry dude!')
        else:
            raise NotImplemented('sorry dude!')
            
        to_add = 0.0
        to_mult = 1.0
        if self._apply_extrap:
            self._apply_extrap = 0
            
            if self.Mlo_extrap:
                p1 = self.pf['sfe_Mfun_lo_par0']
                p2 = self.pf['sfe_Mfun_lo_par1']
                if self.pf['sfe_Mfun_lo'] == 'pl':
                    to_add = self.fstar(z, p1) * (M / p1)**p2
                elif self.pf['sfe_Mfun_lo'] == 'plexp':
                    p3 = self.pf['sfe_Mfun_lo_par2']
                    to_add = self.fstar(z, p1) * (M / p1)**p2 \
                        * np.exp(-p3 / M)
                    
            if self.Mhi_extrap:
                Mexp = self.pf['sfe_Mfun_hi_par0']
                to_mult = np.exp(-M / Mexp)
                
            self._apply_extrap = 1

        f += to_add
        f *= to_mult
    
        return np.minimum(f, self.pf['sfe_ceil'])
