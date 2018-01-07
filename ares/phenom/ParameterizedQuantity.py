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
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

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
        for i in range(8):
            par = 'pq_func_par{}'.format(i)
            
            if par not in self.pf:
                continue
            
            val = self.pf[par]
                
            if not isinstance(val, basestring):
                continue
                
            pq_pars = get_pq_pars(val, self.raw_pf)
            
            PQ = ParameterizedQuantity(**pq_pars)
            
            self._sub_pqs[val] = PQ
            
    @property
    def idnum(self):
        if not hasattr(self, '_idnum'):
            self._idnum = None
        return self._idnum
        
    @idnum.setter
    def idnum(self, value):
        self._idnum = int(value)

    @property
    def func(self):
        return self.pf['pq_func']
    
    @property
    def func_var(self):
        return self.pf['pq_func_var']
    
    @property
    def var_ceil(self):
        if not hasattr(self, '_var_ceil'):
            if 'pq_var_ceil' in self.pf:
                self._var_ceil = self.pf['pq_var_ceil']
            else:
                self._var_ceil = None

        return self._var_ceil        

    @property
    def var_floor(self):
        if not hasattr(self, '_var_floor'):
            if 'pq_var_floor' in self.pf:
                self._var_floor = self.pf['pq_var_floor']
            else:
                self._var_floor = None
    
        return self._var_floor  
    
    @property
    def ceil(self):
        if not hasattr(self, '_ceil'):
            if 'pq_val_ceil' in self.pf:
                self._ceil = self.pf['pq_val_ceil']
            else:
                self._ceil = None
    
        return self._ceil        
        
    @property
    def floor(self):
        if not hasattr(self, '_floor'):
            if 'pq_val_floor' in self.pf:
                if type(self.pf['pq_val_floor']) is str:
                    val = self.pf['pq_val_floor']
                                                                                
                    pq_pars = get_pq_pars(val, self.raw_pf)
                    
                    PQ = ParameterizedQuantity(**pq_pars)
                    
                    self._sub_pqs[val] = PQ
                                        
                    self._floor = PQ
                
                else:
                    self._floor = self.pf['pq_val_floor']
            else:
                self._floor = None
    
        return self._floor
    
    @property
    def pars_list(self):
        if not hasattr(self, '_pars_list'):
            self._pars_list = []
            for i in range(8):
                name = 'pq_func_par{}'.format(i)
                if name in self.pf:
                    self._pars_list.append(self.pf[name])
                else:
                    self._pars_list.append(None)
        return self._pars_list

    def __call__(self, **kwargs):
        return self._call(self.pars_list, **kwargs)

    def _call(self, pars, **kwargs):
        """
        A higher-level version of __call__ that accepts a few more kwargs.
        """

        func = self.func
        
        # Determine independent variables
        var = self.func_var
                
        if var == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[var]

        if self.var_ceil is not None:
            x = np.minimum(x, self.var_ceil)
        if self.var_floor is not None:
            x = np.maximum(x, self.var_floor)

        logx = np.log10(x)

        # Read-in parameters to more convenient names
        # I don't usually use exec, but when I do, it's to do garbage like this
        for i, par in enumerate(pars):

            # It's possible that a parameter will itself be a PQ object.            
            if isinstance(par, basestring):
                         
                # Could call recursively. Implement __getattr__?
                PQ = self._sub_pqs[par]
                                
                val = PQ.__call__(**kwargs)
                
                setattr(self, 'p{}'.format(i), val)
                
            elif type(par) == tuple:
                f, v, mult = par
                
                if isinstance(mult, basestring):
                    m = self.pf[mult]
                else:
                    m = mult
                
                if f in self.deps:
                    val = m * self.deps[f](kwargs[v])
                elif type(f) is FunctionType:
                    val = m * f(kwargs[v])
                else:
                    raise NotImplementedError('help')
                    
                setattr(self, 'p{}'.format(i), val)
                    
            else:
                setattr(self, 'p{}'.format(i), par)

        # Actually execute the function
        if func == 'lognormal':
            f = self.p0 * np.exp(-(logx - self.p1)**2 / 2. / self.p2**2)
        elif func == 'normal':
            f = self.p0 * np.exp(-(x - self.p1)**2 / 2. / self.p2**2)
        elif func == 'pl':
            #print('{0} {1} {2} {3} {4}'.format(x, kwargs['z'], p0, p1, p2))
            f = self.p0 * (x / self.p1)**self.p2
        # 'quadratic_lo' means higher order terms vanish when x << p3
        elif func == 'quadratic_lo':
            f = self.p0 * (1. +  self.p1 * (x / self.p3) + self.p2 * (x / self.p3)**2)
        # 'quadratic_hi' means higher order terms vanish when x >> p3
        elif func in ['quadratic_hi', 'quad']:
            f = self.p0 * (1. +  self.p1 * (self.p3 / x) + self.p2 * (self.p3 / x)**2)
        #elif func == 'cubic_lo':
        #    f = self.p1 * (1. + self.p2 * (x / self.p0) + self.p3 * (x / self.p0)**2)
        #elif func == 'cubic_hi':
        #    f = self.p1 * (1. +  self.p2 * (self.p0 / x) + self.p3 * (self.p0 / x)**2)
        elif func == 'exp':
            f = self.p0 * np.exp((x / self.p1)**self.p2)
        elif func == 'exp_flip':
            f = 1. - self.p0 * np.exp(-(x / self.p1)**self.p2)
        elif func == 'plexp':
            f = self.p0 * (x / self.p1)**self.p2 * np.exp(-(x / self.p3)**self.p4)
        elif func == 'dpl':
            f = 2. * self.p0 / ((x / self.p1)**-self.p2 + (x / self.p1)**-self.p3)
        elif func == 'dpl_arbnorm':
            normcorr = (((self.p4 / self.p1)**-self.p2 + (self.p4 / self.p1)**-self.p3))
            f = self.p0 * normcorr / ((x / self.p1)**-self.p2 + (x / self.p1)**-self.p3)
        elif func == 'ddpl':
            f = 2. * self.p0 / ((x / self.p1)**-self.p2 + (x / self.p1)**-self.p3) \
              + 2. * self.p4 / ((x / self.p5)**-self.p6 + (x / self.p5)**-self.p7)
        elif func == 'ddpl_arbnorm':
            normcorr1 = (((self.p4 / self.p1)**-self.p2 + (self.p4 / self.p1)**-self.p3))
            normcorr2 = (((self.p4 / self.p1)**-self.p2 + (self.p4 / self.p1)**-self.p3))
            f1 = self.p0 * normcorr / ((x / self.p1)**-self.p2 + (x / self.p1)**-self.p3)
            f1 = self.p5 * normcorr / ((x / self.p1)**-self.p2 + (x / self.p1)**-self.p3)
        elif func == 'plsum2':
            f = self.p0 * (x / self.p1)**self.p2 + self.p3 * (x / self.p1)**self.p4
        elif func == 'tanh_abs':
            f = (self.p0 - self.p1) * 0.5 * (np.tanh((self.p2 - x) / self.p3) + 1.) + self.p1
        elif func == 'tanh_rel':
            f = self.p1 * self.p0 * 0.5 * (np.tanh((self.p2 - x) / self.p3) + 1.) + self.p1
        elif func == 'log_tanh_abs':
            f = (self.p0 - self.p1) * 0.5 * (np.tanh((self.p2 - logx) / self.p3) + 1.) + self.p1
        elif func == 'log_tanh_rel':                                       
            f = self.p1 * self.p0 * 0.5 * (np.tanh((self.p2 - logx) / self.p3) + 1.) + self.p1
        elif func == 'rstep':
            if type(x) is np.ndarray:
                lo = x < self.p2
                hi = x >= self.p2

                f = lo * self.p0 * self.p1 + hi * self.p1 
            else:
                if x < p2:
                    f = self.p0 * self.p1
                else:
                    f = self.p1
        elif func == 'astep':

            if type(x) is np.ndarray:
                lo = x < self.p2
                hi = x >= self.p2

                f = lo * self.p0 + hi * self.p1
                
            else:
                if x < self.p2:
                    f = self.p0
                else:
                    f = self.p1
        elif func == 'pwpl':
            if type(x) is np.ndarray:
                lo = x < self.p4
                hi = x >= self.p4

                f = lo * self.p0 * (x / self.p4)**self.p1 + hi * self.p2 * (x / self.p4)**self.p3
            else:
                if x < self.p4:
                    f = self.p0 * (x / self.p4)**self.p1
                else:            
                    f = self.p2 * (x / self.p4)**self.p3
        elif func == 'okamoto':
            assert var == 'Mh'
            f = (1. + (2.**(self.p0 / 3.) - 1.) * (x / self.p1)**-self.p0)**(-3. / self.p0)
        elif func == 'ramp':
            if type(x) is np.ndarray:
                lo = x <= self.p1
                hi = x >= self.p3
                mi = np.logical_and(x > self.p1, x < self.p3)
                
                # ramp slope
                m = (self.p2 - self.p0) / (self.p3 - self.p1)

                f = lo * self.p0 + hi * self.p2 + mi * (self.p0 + m * (x - self.p1))

            else:
                if x <= self.p1:
                    f = self.p0
                elif x >= self.p3:
                    f = self.p1
                else:
                    f = self.p0 + m * (x - self.p1)
        elif func == 'logramp':
            if type(x) is np.ndarray:
                lo = logx <= self.p1
                hi = logx >= self.p3
                mi = np.logical_and(logx > self.p1, logx < self.p3)

                # ramp slope
                alph = np.log10(self.p2 / self.p0) / (self.p3 - self.p1)
                fmid = self.p0 * (x / 10**self.p1)**alph

                f = lo * self.p0 + hi * self.p2 + mi * fmid

            else:
                if logx <= self.p2:
                    f = self.p0
                elif logx >= self.p3:
                    f = self.p1
                else:
                    alph = np.log10(self.p2 / self.p0) / (self.p3 - self.p1)
                    f = (x / 10**self.p1)**alph

        elif func == 'user':
            f = self.pf['pq_func_fun'](**kwargs)
        else:
            raise NotImplementedError('Don\'t know how to treat {!s} function!'.format(func))

        if self.ceil is not None:
            f = np.minimum(f, self.ceil)
        if self.floor is not None:
            if type(self.floor) in [int, float]:
                f = np.maximum(f, self.floor)
            else:
                f = np.maximum(f, self.floor(**kwargs))
        
        return f
              


