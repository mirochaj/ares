"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import numpy as np
from ..util import ParameterFile

def tanh_astep(M, lo, hi, logM0, logdM):
    return (lo - hi) * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi
def tanh_rstep(M, lo, hi, logM0, logdM):
    return hi * lo * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi

#z0 = 9. # arbitrary

Mh_dep_parameters = ['pop_fstar', 'pop_fesc', 'pop_L1500_per_sfr', 
    'pop_Nion', 'pop_Nlw']

class ParameterizedHaloProperty(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
    
    @property
    def Mfunc(self):
        return self.pf['php_Mfun']
    
    @property
    def M_aug(self):
        if not hasattr(self, '_M_aug'):
            self._M_aug = self.pf['php_Mfun_aug'] is not None
        return self._M_aug  

    #def func(self, name):        
    #    if self.pf['php_%s' % name] == 'constant':
    #        func = lambda zz: self.pf['php_%s_par0' % name]
    #    elif self.pf['php_%s' % name] == 'linear_z':
    #        coeff1 = self.pf['php_%s_par0' % name]
    #        coeff2 = self.pf['php_%s_par1' % name]
    #        func = lambda zz: coeff1 + coeff2 * (1. + zz) / (1. + z0)
    #    elif self.pf['php_%s' % name] == 'linear_t':
    #        coeff = self.pf['php_%s_par0' % name]
    #        func = lambda zz: 10**(np.log10(coeff) - 1.5 * (1. + zz) / (1. + z0))
    #    elif self.pf['php_%s' % name] == 'pl':
    #        coeff1 = self.pf['php_%s_par0' % name]
    #        coeff2 = self.pf['php_%s_par1' % name]
    #        func = lambda zz: 10**(np.log10(coeff1) + coeff2 * (1. + zz) / (1. + z0))
    #    elif self.pf['php_%s' % name] == 'poly':
    #        coeff1 = self.pf['php_%s_par0' % name]
    #        coeff2 = self.pf['php_%s_par1' % name]
    #        coeff3 = self.pf['php_%s_par2' % name]
    #        func = lambda zz: 10**(np.log10(coeff1) + coeff2 * (1. + zz) / (1. + z0) \
    #            + coeff3 * ((1. + zz) / (1. + z0))**2)
    #
    #    return func
    #
    @property
    def _apply_extrap(self):
        if not hasattr(self, '_apply_extrap_'):
            #if self.pf['php_Mfun_aug']
            self._apply_extrap_ = 1
        return self._apply_extrap_

    @_apply_extrap.setter
    def _apply_extrap(self, value):
        self._apply_extrap_ = value   

    def __call__(self, z, M):
        """
        Compute the star formation efficiency.
        """

        pars1 = [self.pf['php_Mfun_par%i' % i] for i in range(6)]
        pars2 = []
        
        for i in range(6):
            tmp = []
            for j in range(6):
                name = 'php_Mfun_par%i_par%i' % (i,j)
                if name in self.pf:
                    tmp.append(self.pf[name])
                else:
                    tmp.append(None)
                    
            pars2.append(tmp)

        return self._call(z, M, [pars1, pars2])

    def _call(self, z, M, pars, func=None):
        """
        A higher-level version of __call__ that accepts a few more kwargs.
        """

        if func is None:
            func = self.Mfunc

        logM = np.log10(M)
        
        # [optional] Modify parameters as function of redshift
        pars1, pars2 = pars
        
        # Read-in parameters to more convenient names
        # I don't usually use exec, but when I do, it's to do garbage like this
        for i, par in enumerate(pars1):
            
            # Handle redshift dependencies
            if type(par) == str:
                p = pars2[i]
                if par == 'linear_t':
                    val = p[0] * ((1. + z) / (1. + p[1]))**-1.5
                elif par == 'pl':
                    val = p[0] * ((1. + z) / (1. + p[1]))**p[2]
                
                exec('p%i = val' % i)
            # Otherwise, assume parameter is just a number
            else:
                exec('p%i = par' % i)
                                
        if func == 'lognormal':
            f = p0 * np.exp(-(logM - p1)**2 / 2. / p2**2)    
        elif func == 'pl':
            f = p0 * (M / p1)**p2
        elif func == 'plexp':
            f = p0 * (M / p1)**p2 * np.exp(-M / p3)
        elif func == 'dpl':
            f = 2. * p0 / ((M / p1)**-p2 + (M / p1)**p3)    
        elif func == 'plsum2':
            f = p0 * (M / p1)**p2 + p3 * (M / p1)**p4
        elif func == 'tanh_abs':
            return tanh_astep(M, p0, p1, p2, p3)
        elif func == 'tanh_rel':
            return tanh_rstep(M, p0, p1, p2, p3)
        elif func == 'rstep':
            if type(M) is np.ndarray:
                lo = M <= p2
                hi = M > p2
        
                return lo * p0 * p1 + hi * p1 
            else:
                if M <= p2:
                    return p0 * p1
                else:
                    return p1
        elif func == 'astep':
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
            if type(M) is np.ndarray:
                lo = M <= p4
                hi = M > p4

                return lo * p0 * (M / p4)**p1 \
                     + hi * p2 * (M / p4)**p3
            else:
                if M <= p4:
                    return p0 * (M / p1)**p2
                else:
                    return p3 * (M / p1)**p4
        elif func == 'okamoto':
            f = (1. + (2.**(p0 / 3.) - 1.) * (M / p1)**-p0)**(-3. / p0)
        elif func == 'user':
            f = self.pf['php_Mfun_fun'](z, M)
        else:
            raise NotImplemented('sorry dude!')

        # Add or multiply to main function.
        if self.M_aug and self._apply_extrap:
            self._apply_extrap = 0
            
            p = [self.pf['php_Mfun_aug_par%i' % i] for i in range(6)]
            aug = self._call(z, M, [p, None], self.pf['php_Mfun_aug'])
            
            if self.pf['php_Mfun_aug_meth'] == 'multiply':
                f *= aug
            else:
                f += aug

            self._apply_extrap = 1    
            
        if self.pf['php_ceil'] is not None:
            f = np.minimum(f, self.pf['php_ceil'])
        if self.pf['php_floor'] is not None:
            f = np.maximum(f, self.pf['php_floor'])
        
        f *= self.pf['php_boost']
        f /= self.pf['php_iboost']
        
        return f
              
        
