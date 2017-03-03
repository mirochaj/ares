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
from ..util.ParameterFile import get_pq_pars

def tanh_astep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return (lo - hi) * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi
def tanh_rstep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return hi * lo * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi

func_options = \
{
 'pl': 'p[0] * (x / p[1])**p[2]',
 'exp': 'p[0] * exp(-(x / p[1])**p[2])',
 'exp_flip': 'p[0] * exp(-(x / p[1])**p[2])', 
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

class ParameterizedQuantity(object):
    def __init__(self, deps={}, raw_pf={}, **kwargs):
        self.pf = ParameterFile(**kwargs) # why all?
        
        # The only reason this is here is because
        # occasionally we need to pass in pop_Mmin
        self.deps = deps
        self.raw_pf = raw_pf
        
        self._set_sub_pqs()
        
    def _set_sub_pqs(self):
        """
        Determine if there are any nested ParameterizedQuantity objects.
        """
        
        self._sub_pqs = {}
        for i in range(6):
            par = 'pq_func_par%i' % i
            val = self.pf[par]
            if type(val) != str:
                continue
                
            pq_pars = get_pq_pars(val, self.raw_pf)
            
            PQ = ParameterizedQuantity(**pq_pars)
            
            self._sub_pqs[val] = PQ

    @property
    def func(self):
        return self.pf['pq_func']
    
    @property
    def func_var(self):
        return self.pf['pq_func_var']

    def __call__(self, **kwargs):
        """
        Compute the star formation efficiency.
        """

        pars = [self.pf['pq_func_par%i' % i] for i in range(6)]

        return self._call(pars, **kwargs)

    def _call(self, pars, **kwargs):
        """
        A higher-level version of __call__ that accepts a few more kwargs.
        """


        func = self.func
        
        # Determine independent variables
        var = self.pf['pq_func_var']
        
        if var == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[var]

        if self.pf['pq_var_ceil'] is not None:
            x = np.minimum(x, self.pf['pq_var_ceil'])
        if self.pf['pq_var_floor'] is not None:
            x = np.maximum(x, self.pf['pq_var_floor'])
        
        logx = np.log10(x)
        
        # [optional] Modify parameters as function of redshift
        #pars1, = pars
        
        # Read-in parameters to more convenient names
        # I don't usually use exec, but when I do, it's to do garbage like this
        for i, par in enumerate(pars):
            
            # It's possible that a parameter will itself be a PQ object.            
            if type(par) == str:
                _pq_pars = get_pq_pars(par, self.raw_pf)
                                
                # Could call recursively. Implement __getattr__?
                PQ = self._sub_pqs[par]
                                
                val = PQ.__call__(**kwargs)
                
                exec('p%i = val' % i)
                
            elif type(par) == tuple:
                f, v, mult = par
                
                if type(mult) is str:
                    m = self.pf[mult]
                else:
                    m = mult
                
                if f in self.deps:
                    val = m * self.deps[f](kwargs[v])
                elif type(f) is FunctionType:
                    val = m * f(kwargs[v])    
                else:
                    raise NotImplementedError('help')
                    
                exec('p%i = val' % i)    
                    
            else:
                exec('p%i = par' % i)
            
        # Actually execute the function                    
        if func == 'lognormal':
            f = p0 * np.exp(-(logx - p1)**2 / 2. / p2**2)   
        elif func == 'normal':
            f = p0 * np.exp(-(x - p1)**2 / 2. / p2**2)
        elif func == 'pl':
            f = p0 * (x / p1)**p2
        # 'quadratic_lo' means higher order terms vanish when x << p0
        elif func == 'quadratic_lo':
            f = p1 * (1. +  p2 * (x / p0) + p3 * (x / p0)**2)
        # 'quadratic_hi' means higher order terms vanish when x >> p0
        elif func == 'quadratic_hi':
            f = p1 * (1. +  p2 * (p0 / x) + p3 * (p0 / x)**2)
        #elif func == 'cubic_lo':
        #    f = p1 * (1. +  p2 * (x / p0) + p3 * (x / p0)**2)
        #elif func == 'cubic_hi':
        #    f = p1 * (1. +  p2 * (p0 / x) + p3 * (p0 / x)**2)    
        elif func == 'exp':
            f = p0 * np.exp(-(x / p1)**p2)
        elif func == 'exp_flip':
            f = 1. - p0 * np.exp(-(x / p1)**p2)
        elif func == 'plexp':
            f = p0 * (x / p1)**p2 * np.exp(-(x / p3)**p4)
        elif func == 'dpl':
            f = 2. * p0 / ((x / p1)**-p2 + (x / p1)**-p3)    
        elif func == 'dpl_arbnorm':
            normcorr = (((p4 / p1)**-p2 + (p4 / p1)**-p3))
            f = p0 * normcorr / ((x / p1)**-p2 + (x / p1)**-p3)
        elif func == 'plsum2':
            f = p0 * (x / p1)**p2 + p3 * (x / p1)**p4
        elif func == 'tanh_abs':
            f = (p0 - p1) * 0.5 * (np.tanh((p2 - x) / p3) + 1.) + p1
        elif func == 'tanh_rel':
            f = p1 * p0 * 0.5 * (np.tanh((p2 - x) / p3) + 1.) + p1  
        elif func == 'log_tanh_abs':
            f = (p0 - p1) * 0.5 * (np.tanh((p2 - logx) / p3) + 1.) + p1
        elif func == 'log_tanh_rel':                                        
            f = p1 * p0 * 0.5 * (np.tanh((p2 - logx) / p3) + 1.) + p1
        elif func == 'rstep':
            if type(x) is np.ndarray:
                lo = x < p2
                hi = x >= p2
        
                f = lo * p0 * p1 + hi * p1 
            else:
                if x < p2:
                    f = p0 * p1
                else:
                    f = p1
        elif func == 'astep':
            
            if type(x) is np.ndarray:
                lo = x < p2
                hi = x >= p2

                f = lo * p0 + hi * p1
                
            else:
                if x < p2:
                    f = p0
                else:
                    f = p1      
        elif func == 'pwpl':
            if type(x) is np.ndarray:
                lo = x < p4
                hi = x >= p4

                f = lo * p0 * (x / p4)**p1 + hi * p2 * (x / p4)**p3
            else:
                if x < p4:
                    f = p0 * (x / p4)**p1
                else:            
                    f = p2 * (x / p4)**p3
        elif func == 'okamoto':
            assert var == 'Mh'
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
        #    f = self.pf['pq_func_fun'](z, M)
        else:
            raise NotImplementedError('Don\'t know how to treat %s function!' % func)

        if self.pf['pq_val_ceil'] is not None:
            f = np.minimum(f, self.pf['pq_val_ceil'])
        if self.pf['pq_val_floor'] is not None:
            f = np.maximum(f, self.pf['pq_val_floor'])
        
        return f
              


