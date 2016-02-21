"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import numpy as np
from .ParameterFile import ParameterFile

def tanh_astep(M, lo, hi, logM0, logdM):
    return (lo - hi) * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi
def tanh_rstep(M, lo, hi, logM0, logdM):
    return hi * lo * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi

z0 = 9. # arbitrary

Mh_dep_parameters = ['pop_fstar', 'pop_fesc', 'pop_L1500_per_sfr', 
    'pop_Nion', 'pop_Nlw']

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
    def M_aug(self):
        if not hasattr(self, '_M_aug'):
            self._M_aug = self.pf['php_Mfun_aug'] is not None
        return self._M_aug  
          
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

        pars = [self.pf['php_Mfun_par%i' % i] for i in range(6)]
        #lpars = [self.pf['php_Mfun_lo_par%i' % i] for i in range(4)]
        #hpars = [self.pf['php_Mfun_hi_par%i' % i] for i in range(4)]

        return self._call(z, M, pars)

    def _call(self, z, M, pars, func=None):

        if func is None:
            func = self.Mfunc

        logM = np.log10(M)
                
        if func == 'lognormal':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
            f = p0 * np.exp(-(logM - p1)**2 / 2. / p2**2)    
            #f = self.fpeak(z) * np.exp(-(logM - np.log10(self.Mpeak(z)))**2 \
            #    / 2. / self.sigma(z)**2)
        elif func == 'pl':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
            f = p0 * (M / p1)**p2
        elif func == 'plexp':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
            f = p0 * (M / 1e10)**p1 * np.exp(-M / p2)
        elif func == 'dpl':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            f = 2. * p0 / ((M / p1)**-p2 + (M / p1)**p3)    
        elif func == 'plsum2':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            f = p0 * (M / 1e10)**p1 + p2 * (M / 1e10)**p3
        elif func == 'rstep':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
            
            if type(M) is np.ndarray:
                lo = M <= p2
                hi = M > p2
                
                return lo * p0 * p1 + hi * p1 
            else:
                if M <= p2:
                    return p0 * p1
                else:
                    return p1
        elif func == 'tanh_abs':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            return tanh_astep(M, p0, p1, p2, p3)
        elif func == 'tanh_rel':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            return tanh_rstep(M, p0, p1, p2, p3)    
            
        elif func == 'astep':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]
        
            if type(M) is np.ndarray:
                lo = M <= p2
                hi = M > p2
        
                return lo * p0 + hi * p1 
            else:
                if M <= p2:
                    return p0
                else:
                    return p1            
        elif func == 'pwpl':
            p0 = pars[0]; p1 = pars[1]; p2 = pars[2]; p3 = pars[3]
            p4 = pars[4]; p5 = pars[5]
            
            if type(M) is np.ndarray:
                lo = M <= p4
                hi = M > p4
                
                return lo * p0 * (M / p4)**p1 \
                     + hi * p2 * (M / p4)**p3
            else:
                if M <= p4:
                    return p0 * (M / 1e10)**p1
                else:
                    return p2 * (M / 1e10)**p3
        elif func == 'okamoto':
            p0 = pars[0]; p1 = pars[1]
            f = (1. + (2.**(p0 / 3.) - 1.) * (M / p1)**-p0)**(-3. / p0)
        elif func == 'user':
            f = self.pf['php_Mfun_fun'](z, M)

        else:
            raise NotImplemented('sorry dude!')

        # Add or multiply to main function.
        if self.M_aug and self._apply_extrap:
            self._apply_extrap = 0
            
            p = [self.pf['php_Mfun_aug_par%i' % i] for i in range(6)]
            aug = self._call(z, M, p, self.pf['php_Mfun_aug'])
            
            if self.pf['php_Mfun_aug_meth'] == 'multiply':
                f *= aug
            else:
                f += aug
                                
            self._apply_extrap = 1    
            
        if self.pf['php_ceil'] is not None:
            f = np.minimum(f, self.pf['php_ceil'])
        if self.pf['php_floor'] is not None:
            f = np.maximum(f, self.pf['php_floor'])
        
        return f
              
        
