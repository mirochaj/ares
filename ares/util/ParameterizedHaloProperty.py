"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import numpy as np
from .ParameterFile import ParameterFile

z0 = 9. # arbitrary

class ParameterizedHaloProperty(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
    
    @property
    def Mfunc(self):
        return self.pf['php_Mfun']
    
    @property
    def zfunc(self):
        return self.pf['php_zfun']
        
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
            self._Mlo_extrap = self.pf['php_Mfun_lo'] is not None
        return self._Mlo_extrap
    @property
    def Mhi_extrap(self):
        if not hasattr(self, '_Mhi_extrap'):
            self._Mhi_extrap = self.pf['php_Mfun_hi'] is not None
        return self._Mhi_extrap   
    
    def func(self, name):        
        if self.pf['php_%s' % name] == 'constant':
            func = lambda zz: self.pf['php_%s_par0' % name]
        elif self.pf['php_%s' % name] == 'linear_z':
            coeff1 = self.pf['php_%s_par0' % name]
            coeff2 = self.pf['php_%s_par1' % name]
            func = lambda zz: coeff1 + coeff2 * (1. + zz) / (1. + z0)
        elif self.pf['php_%s' % name] == 'linear_t':
            coeff = self.pf['php_%s_par0' % name]
            func = lambda zz: 10**(np.log10(coeff) - 1.5 * (1. + zz) / (1. + z0))
        elif self.pf['php_%s' % name] == 'pl':
            coeff1 = self.pf['php_%s_par0' % name]
            coeff2 = self.pf['php_%s_par1' % name]
            func = lambda zz: 10**(np.log10(coeff1) + coeff2 * (1. + zz) / (1. + z0))
        elif self.pf['php_%s' % name] == 'poly':
            coeff1 = self.pf['php_%s_par0' % name]
            coeff2 = self.pf['php_%s_par1' % name]
            coeff3 = self.pf['php_%s_par2' % name]
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

    def __call__(self, z, M):
        """
        Compute the star formation efficiency.
        """

        pars = [self.pf['php_Mfun_par%i' % i] for i in range(4)]
        lpars = [self.pf['php_Mfun_lo_par%i' % i] for i in range(4)]
        hpars = [self.pf['php_Mfun_hi_par%i' % i] for i in range(4)]

        return self._call(z, M, pars, lpars, hpars)

    def _call(self, z, M, pars, lopars=None, hipars=None):

        logM = np.log10(M)

        if self.Mfunc == 'lognormal':            
            f = self.fpeak(z) * np.exp(-(logM - np.log10(self.Mpeak(z)))**2 \
                / 2. / self.sigma(z)**2)
        elif self.Mfunc == 'pl':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
            f = p0 * (M / p1)**p2
        elif self.Mfunc == 'dpl':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            f = 2. * p0 / ((M / p1)**-p2 + (M / p1)**p3)    
        elif self.Mfunc == 'poly':
            raise NotImplemented('sorry dude!')
        else:
            raise NotImplemented('sorry dude!')
    
        to_add = 0.0
        to_mult = 1.0
        if self._apply_extrap:
            self._apply_extrap = 0

            if self.Mlo_extrap:
                p0 = self.pf['php_Mfun_lo_par0']
                p1 = self.pf['php_Mfun_lo_par1']
                if self.pf['php_Mfun_lo'] == 'pl':
                    to_add = p0 * (M / 1e10)**p1
                elif self.pf['php_Mfun_lo'] == 'plexp':
                    p2 = self.pf['php_Mfun_lo_par2']
                    to_add = p0 * (M / 1e10)**p1 * np.exp(-M / p2)
                elif self.pf['php_Mfun_lo'] == 'dpl':
                    p2 = self.pf['php_Mfun_lo_par2']
                    p3 = self.pf['php_Mfun_lo_par3']
                    to_add = 2 * p1 / ((M / p2)**-p3 + (M / p2)**p4)

            if self.Mhi_extrap:
                if self.pf['php_Mfun_hi'] == 'exp':
                    Mexp = self.pf['php_Mfun_hi_par0']
                    to_mult = np.exp(-M / Mexp)
                elif self.pf['php_Mfun_hi'] == 'pl':    
                    Mt = self.pf['php_Mfun_hi_par0']
                    dM = self.pf['php_Mfun_hi_par1']
                    to_mult = 1. - np.tanh((M - Mt) / dM)   
                
            self._apply_extrap = 1

        f += to_add
        f *= to_mult
    
        return np.minimum(f, self.pf['php_ceil'])
              
        
