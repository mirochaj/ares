"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import numpy as np
from types import FunctionType
from ..util import ParameterFile

def tanh_astep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return (lo - hi) * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi
def tanh_rstep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return hi * lo * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi

Mh_dep_parameters = ['pop_fstar', 'pop_fesc', 'pop_L1600_per_sfr', 
    'pop_Nion', 'pop_Nlw', 'pop_fesc_LW', 'pop_yield']
    
func_options = \
{
 'pl': 'p[0] * (x / p[1])**p[2]',
 'dpl': 'p[0] / ((x / p[1])**-p[2] + (x / p[1])**-p[3])',
 'dpl_arbnorm': 'p[0](p[4]) / ((x / p[1])**-p[2] + (x / p[1])**-p[3])',
 'pwpl': 'p[0] * (x / p[4])**p[1] if x <= p[4] else p[2] * (x / p[4])**p[3]',
 'plexp': 'p[0] * (x / p[1])**p[2] * np.exp(-x / p[3])',
 'lognormal': 'p[0] * np.exp(-(logx - p[1])**2 / 2. / p[2]**2)',
 'astep': 'p0 if x <= p1 else p2',
 'rstep': 'p0 * p2 if x <= p1 else p2',
 'plsum': 'p[0] * (x / p[1])**p[2] + p[3] * (x / p[4])**p5',
 'ramp': 'p0 if x <= p1, p2 if x >= p3, linear in between',
}

class ParameterizedHaloProperty(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
    
    @property
    def func(self):
        return self.pf['php_func']
    
    @property
    def func_vars(self):
        return self.pf['php_vars']

    @property
    def faux(self):
        if not hasattr(self, '_faux'):
            self._faux = False
            for faux_id in ['', '_A', '_B']:
                                
                if self.pf['php_faux%s' % faux_id] is None:
                    continue
                
                self._faux = True
                break
                
        return self._faux

    @property
    def _apply_extrap(self):
        if not hasattr(self, '_apply_extrap_'):
            self._apply_extrap_ = 1
        return self._apply_extrap_

    @_apply_extrap.setter
    def _apply_extrap(self, value):
        self._apply_extrap_ = value

    def __call__(self, z, M=None, func=None):
        """
        Compute the star formation efficiency.
        """

        pars1 = [self.pf['php_func_par%i' % i] for i in range(6)]
        pars2 = []

        for i in range(6):
            tmp = []
            for j in range(6):
                name = 'php_func_par%i_par%i' % (i,j)
                if name in self.pf:
                    tmp.append(self.pf[name])
                else:
                    tmp.append(None)
        
            pars2.append(tmp)
        
        return self._call(z, M, [pars1, pars2])

    def _call(self, z, M, pars, func=None, faux_id=''):
        """
        A higher-level version of __call__ that accepts a few more kwargs.
        """

        if func is None:
            func = self.func
            s = 'func' 
        # Otherwise, assume it's the auxilary function
        else:
            s = 'faux%s' % faux_id

        # Determine independent variables
        var = self.pf['php_%s_var' % s].lower()
        if var == 'mass':
            x = M
        elif (var == 'redshift') or (var == 'z'):
            x = z
        elif var == '1+z':
            x = 1. + z
        else:
            raise ValueError('Unrecognized func_var \'%s\'.' % var)    
        
        logx = np.log10(x)
        
        # [optional] Modify parameters as function of redshift
        pars1, pars2 = pars
        
        # Read-in parameters to more convenient names
        # I don't usually use exec, but when I do, it's to do garbage like this
        for i, par in enumerate(pars1):
            
            if type(par) == str:
                
                # Parameters that are...parameterized! Things are nested, i.e,
                # fstar is not necessarily separable.
                
                assert par == 'pl', 'Only support for PL extensions.'
                                
                p = pars2[i]
                val = p[0] * ((1. + z) / p[1])**p[2]
                                
                exec('p%i = val' % i)
            elif type(par) == tuple:
                f, v = par
                                
                if v == 'z':
                    val = f(z)
                elif v == 'mass':
                    val = f(M)
                else:
                    raise NotImplementedError('help!')
                
                exec('p%i = val' % i)    
                    
            else:
                exec('p%i = par' % i)
            
        # Actually execute the function                    
        if func == 'lognormal':
            f = p0 * np.exp(-(logx - p1)**2 / 2. / p2**2)    
        elif func == 'pl':
            f = p0 * (x / p1)**p2
        elif func == 'plexp':
            f = p0 * (x / p1)**p2 * np.exp(-x / p3)
        elif func == 'dpl':
            f = 2. * p0 / ((x / p1)**-p2 + (x / p1)**-p3)    
        elif func == 'dpl_arbnorm':
            normcorr = (((p4 / p1)**-p2 + (p4 / p1)**-p3))
            f = p0 * normcorr / ((x / p1)**-p2 + (x / p1)**-p3)
        elif func == 'plsum2':
            f = p0 * (x / p1)**p2 + p3 * (x / p1)**p4
        elif func == 'tanh_abs':
            f = tanh_astep(x, p0, p1, p2, p3)
        elif func == 'tanh_rel':
            f = tanh_rstep(x, p0, p1, p2, p3)
        elif func == 'rstep':
            if type(x) is np.ndarray:
                lo = x <= p2
                hi = x > p2
        
                f = lo * p0 * p1 + hi * p1 
            else:
                if x <= p2:
                    f = p0 * p1
                else:
                    f = p1
        elif func == 'astep':
                        
            if type(x) is np.ndarray:
                lo = x <= p2
                hi = x > p2

                f = lo * p0 + hi * p1 
            else:
                if x <= p2:
                    f = p0
                else:
                    f = p1      
        elif func == 'pwpl':
            if type(x) is np.ndarray:
                lo = x <= p4
                hi = x > p4

                f = lo * p0 * (x / p4)**p1 + hi * p2 * (x / p4)**p3
            else:
                if x <= p4:
                    f = p0 * (x / p4)**p1
                else:            
                    f = p2 * (x / p4)**p3
        elif func == 'okamoto':
            assert var == 'mass'
            f = (1. + (2.**(p0 / 3.) - 1.) * (x / p1)**-p0)**(-3. / p0)
        elif func == 'ramp':
            if type(x) is np.ndarray:
                lo = x <= p1
                hi = x >= p3
                mi = np.logical_and(x > p1, x < p3)
                
                # ramp slope
                m = (p2 - p0) / (p3 - p1)

                f = lo * p0 + hi * p2 + mi * (p0 + m * (x - p1))
                            
            else:
                if x <= p1:
                    f = p0
                elif x >= p3:
                    f = p1
                else:
                    f = p0 + m * (x - p1)
        elif func == 'logramp':
            if type(x) is np.ndarray:
                lo = logx <= p1
                hi = logx >= p3
                mi = np.logical_and(logx > p1, logx < p3)
        
                # ramp slope
                alph = np.log10(p2 / p0) / (p3 - p1)
                fmid = p0 * (x / 10**p1)**alph
                
                f = lo * p0 + hi * p2 + mi * fmid
        
            else:
                if logx <= p2:
                    f = p0
                elif logx >= p3:
                    f = p1
                else:
                    alph = np.log10(p2 / p0) / (p3 - p1)
                    f = (x / 10**p1)**alph

        #elif func == 'user':
        #    f = self.pf['php_func_fun'](z, M)
        else:
            raise NotImplementedError('Don\'t know how to treat %s function!' % func)

        # Add or multiply to main function.
        if self.faux and self._apply_extrap:
            self._apply_extrap = 0

            for k, faux_id in enumerate(['', '_A', '_B']):
                                
                if self.pf['php_faux%s' % faux_id] is None:
                    continue
                                
                p = [self.pf['php_faux%s_par%s' % (faux_id, i)] for i in range(6)]
                aug = self._call(z, M, [p,None], self.pf['php_faux%s' % faux_id], faux_id)

                if self.pf['php_faux%s_meth' % faux_id] == 'multiply':
                    f *= aug
                elif self.pf['php_faux%s_meth' % faux_id] == 'add':
                    f += aug
                else:    
                    raise NotImplemented('Unknown faux_meth \'%s\'' % self.pf['%s_meth' % par_pre])

            self._apply_extrap = 1 

        # Only apply floor/ceil after auxiliary function has been applied
        if self._apply_extrap:

            if self.pf['php_ceil'] is not None:
                f = np.minimum(f, self.pf['php_ceil'])
            if self.pf['php_floor'] is not None:
                f = np.maximum(f, self.pf['php_floor'])

        return f
              


