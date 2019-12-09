"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description: 

"""

import gc
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
 'exp': 'p[0] * exp((x / p[1])**p[2])',
 'exp-m': 'p[0] * exp(-(x / p[1])**p[2])',
 'exp_flip': 'p[0] * exp(-(x / p[1])**p[2])', 
 'dpl': 'p[0] / ((x / p[1])**-p[2] + (x / p[1])**-p[3])',
 'dpl_arbnorm': 'p[0] * (p[4]) / ((x / p[1])**-p[2] + (x / p[1])**-p[3])',
 'pwpl': 'p[0] * (x / p[4])**p[1] if x <= p[4] else p[2] * (x / p[4])**p[3]',
 'plexp': 'p[0] * (x / p[1])**p[2] * np.exp(-x / p[3])',
 'lognormal': 'p[0] * np.exp(-(logx - p[1])**2 / 2. / p[2]**2)',
 'astep': 'p0 if x <= p1 else p2',
 'rstep': 'p0 * p2 if x <= p1 else p2',
 'plsum': 'p[0] * (x / p[1])**p[2] + p[3] * (x / p[4])**p5',
 'ramp': 'p0 if x <= p1, p2 if x >= p3, linear in between',
}

Np_max = 15

optional_kwargs = 'pq_val_ceil', 'pq_val_floor', 'pq_var_ceil', 'pq_var_floor'
numeric_types = [int, float, np.int, np.int64, np.float64]    
    
class BasePQ(object): 
    def __init__(self, **kwargs):
        self.args = []
        for i in range(Np_max):
            name = 'pq_func_par{}'.format(i)
            if name not in kwargs:
                continue
                
            self.args.append(kwargs[name])    

        self.x = kwargs['pq_func_var']
        
        for key in optional_kwargs:
            if key not in kwargs:
                setattr(self, key[3:], None)
            else:
                setattr(self, key[3:], kwargs[key])
                
        if 'pq_func_var2' in kwargs:
            self.t = kwargs['pq_func_var2']
        
class PowerLaw(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        return self.args[0] * (x / self.args[1])**self.args[2]

class PowerLawEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
        
        p0 = self.args[0] * (t / self.args[3])**self.args[4]
        
        return p0 * (x / self.args[1])**self.args[2]


class Exponential(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        return self.args[0] * np.exp((x / self.args[1])**self.args[2])

class ExponentialInverse(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        return self.args[0] * np.exp(-(x / self.args[1])**self.args[2])
    
class Normal(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        return self.args[0] * np.exp(-(x - self.args[1])**2 
            / 2. / self.args[2]**2)

class LogNormal(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logx = np.log10(x)
        return self.args[0] * np.exp(-(logx - self.args[1])**2 
            / 2. / self.args[2]**2)
    
class PiecewisePowerLaw(BasePQ):    
    def __call__(self, **kwargs):
        x = kwargs[self.x]
    
        lo = x < self.args[4]
        hi = x >= self.args[4]

        y = lo * self.args[0] * (x / self.args[4])**self.args[1] \
          + hi * self.args[2] * (x / self.args[4])**self.args[3]

        return y
        
class Ramp(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        # ramp slope
        m = (self.args[2] - self.args[0]) / (self.args[3] - self.args[1])
        
        lo = x <= self.args[1]
        hi = x >= self.args[3]
        mi = np.logical_and(x > self.args[1], x < self.args[3])
        
        y = lo * self.args[0] \
          + hi * self.args[2] + mi * (self.args[0] + m * (x - self.args[1]))

        return y
        
class LogRamp(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logx = np.log10(x)
        
        # ramp slope
        alph = np.log10(self.args[2] / self.args[0]) \
             / (self.args[3] - self.args[1])
                        
        lo = logx <= self.args[1]
        hi = logx >= self.args[3]
        mi = np.logical_and(logx > self.args[1], logx < self.args[3])
        
        fmid = self.args[0] * (x / 10**self.args[1])**alph
        
        y = lo * self.args[0] + hi * self.args[2] + mi * fmid
         
        return y
        
class TanhAbs(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        step = (self.args[0] - self.args[1]) * 0.5        
        y = self.args[1] \
          + step * (np.tanh((self.args[2] - x) / self.args[3]) + 1.)
        return y

class TanhRel(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        y = self.args[1] \
          + self.args[1] * self.args[0] * 0.5 \
          * (np.tanh((self.args[2] - x) / self.args[3]) + 1.)
        
        return y
    
class LogTanhAbs(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logx = np.log10(x)
        
        step = (self.args[0] - self.args[1]) * 0.5        
        y = self.args[1] \
          + step * (np.tanh((self.args[2] - logx) / self.args[3]) + 1.)
        return y

class LogTanhRel(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logx = np.log10(x)
        
        y = self.args[1] \
          + self.args[1] * self.args[0] * 0.5 \
          * (np.tanh((self.args[2] - logx) / self.args[3]) + 1.)
        
        return y        

class StepRel(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        lo = x < self.args[2]
        hi = x >= self.args[2]

        y = lo * self.args[0] * self.args[1] + hi * self.args[1]

        return y 
    
class StepAbs(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        lo = x < self.args[2]
        hi = x >= self.args[2]

        y = lo * self.args[0] + hi * self.args[1] 

        return y


class DoublePowerLawPeakNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)        
        y *= 2. * self.args[0]

        return y
        
class DoublePowerLaw(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))
        
        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)        
        y *= normcorr * self.args[0]

        return y

class DoublePowerLawExtended(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))
        
        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)        
        y *= normcorr * self.args[0]
        
        y *= (1. + (x / self.args[5])**self.args[6])**self.args[7]

        return y        

class DoublePowerLawEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        # Normalization evolves
        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))
        
        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)      
        
        if self.t == '1+z':  
            y *= normcorr * self.args[0] \
                * ((1. + kwargs['z']) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        return y

class DoublePowerLawEvolvingPeak(BasePQ):
    def __call__(self, **kwargs):     
        x = kwargs[self.x]
        
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
        
        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[6]
        
        # Normalization evolves
        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))
        
        # This is to conserve memory.
        y  = (x / p1)**-self.args[2]
        y += (x / p1)**-self.args[3]
        np.divide(1., y, out=y)      
        
        y *= normcorr * self.args[0]
        return y


class DoublePowerLawEvolvingNormPeak(BasePQ):
    def __call__(self, **kwargs):     
        x = kwargs[self.x]
                
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
            
        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]    
        
        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))
        
        # This is to conserve memory.
        xx = x / p1
        y  = xx**-self.args[2]
        y += xx**-self.args[3]
        np.divide(1., y, out=y)      
        
        if self.t == '1+z':  
            y *= normcorr * self.args[0] \
                * ((1. + kwargs['z']) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        return y

class DoublePowerLawEvolvingNormPeakSlope(BasePQ):
    def __call__(self, **kwargs):     
        x = kwargs[self.x]
                
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
            
        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]    
        
        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))
        
        s1 = self.args[2] * (t / self.args[5])**self.args[8]
        s2 = self.args[3] * (t / self.args[5])**self.args[9]
        
        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)      
        
        if self.t == '1+z':  
            y *= normcorr * self.args[0] \
                * ((1. + kwargs['z']) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        return y

class DoublePowerLawEvolvingNormPeakSlopeFloor(BasePQ):
    def __call__(self, **kwargs):     
        x = kwargs[self.x]
                
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
            
        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]    
        
        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))
        
        s1 = self.args[2] * (t / self.args[5])**self.args[8]
        s2 = self.args[3] * (t / self.args[5])**self.args[9]
        
        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)      
        
        if self.t == '1+z':  
            y *= normcorr * self.args[0] \
                * (t / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (t / self.args[5])**self.args[6]


        floor = self.args[10] * (t / self.args[5])**self.args[11]    

        return np.maximum(y, floor)

        
class Okamoto(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        y = (1. + (2.**(self.args[0] / 3.) - 1.) \
          * (x / self.args[1])**-self.args[0])**(-3. / self.args[0])
         
        return y

class OkamotoEvolving(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
        
        p0 = self.args[0] * (t / self.args[2])**self.args[3]
        p1 = self.args[1] * (t / self.args[2])**self.args[4]
            
        y = (1. + (2.**(p0 / 3.) - 1.) * (x / p1)**-p0)**(-3. / p0)
         
        return y

class Schechter(BasePQ):  
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        if self.x.lower() in ['mags', 'muv', 'mag']:
            y = 0.4 * np.log(10.) * 10**self.args[0] \
                * (10**(0.4 * (self.args[1] - x)))**(self.args[2] + 1.) \
                * np.exp(-10**(0.4 * (self.args[1] - x))) 
        else:
            y = 10**self.args[0] * (x / self.args[1])**self.args[2] \
              * np.exp(-(x / self.args[1])) / self.args[1]
        
        return y

class SchechterEvolving(BasePQ):  
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        
        if self.t == '1+z':
            t = 1. + kwargs['z']
        else:
            t = kwargs[self.t]
        
        p0 = 10**(self.args[0] + self.args[4] * (t - self.args[3]))
        p1 = self.args[1] + self.args[5] * (t - self.args[3])
        p2 = self.args[2] + self.args[6] * (t - self.args[3])
        
        if self.x.lower() in ['mags', 'muv', 'mag']:
            y = 0.4 * np.log(10.) * p0 \
                * (10**(0.4 * (p1 - x)))**(p2 + 1.) \
                * np.exp(-10**(0.4 * (p1 - x)))
        else:
            y = p0 * (x / p1)**p2 * np.exp(-(x / p1)) / p1
        
        return y
        
class Linear(BasePQ):            
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        y = self.args[0] + self.args[2] * (x - self.args[1])
        return y
        
class LogLinear(BasePQ):            
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logy = self.args[0] + self.args[2] * (x - self.args[1])
        y = 10**logy
        return y

class ParameterizedQuantity(object):
    def __init__(self, **kwargs):
        if kwargs['pq_func'] == 'pl':
            self.func = PowerLaw(**kwargs)
        elif kwargs['pq_func'] == 'pl_evolN':
            self.func = PowerLawEvolvingNorm(**kwargs)
        elif kwargs['pq_func'] in ['dpl', 'dpl_arbnorm']:
            self.func = DoublePowerLaw(**kwargs)
        elif kwargs['pq_func'] == 'dplx':
            self.func = DoublePowerLawExtended(**kwargs)
        elif kwargs['pq_func'] in ['dpl_normP']:
            self.func = DoublePowerLawPeakNorm(**kwargs)    
        elif kwargs['pq_func'] == 'dpl_evolN':
            self.func = DoublePowerLawEvolvingNorm(**kwargs)
        elif kwargs['pq_func'] == 'dpl_evolP':
            self.func = DoublePowerLawEvolvingPeak(**kwargs)    
        elif kwargs['pq_func'] == 'dpl_evolNP':
            self.func = DoublePowerLawEvolvingNormPeak(**kwargs)  
        elif kwargs['pq_func'] == 'dpl_evolNPS':
            self.func = DoublePowerLawEvolvingNormPeakSlope(**kwargs)
        elif kwargs['pq_func'] == 'dpl_evolNPSF':    
            self.func = DoublePowerLawEvolvingNormPeakSlopeFloor(**kwargs)
        elif kwargs['pq_func'] == 'exp':
            self.func = Exponential(**kwargs)  
        elif kwargs['pq_func'] == 'exp-':
            self.func = ExponentialInverse(**kwargs)
        elif kwargs['pq_func'] == 'pwpl':
            self.func = PiecewisePowerLaw(**kwargs) 
        elif kwargs['pq_func'] == 'ramp':
            self.func = Ramp(**kwargs)   
        elif kwargs['pq_func'] == 'logramp':
            self.func = LogRamp(**kwargs) 
        elif kwargs['pq_func'] == 'tanh_abs':
            self.func = TanhAbs(**kwargs)
        elif kwargs['pq_func'] == 'tanh_rel':
            self.func = TanhRel(**kwargs)
        elif kwargs['pq_func'] == 'step_abs':
            self.func = StepAbs(**kwargs)
        elif kwargs['pq_func'] == 'step_rel':
            self.func = StepRel(**kwargs)
        elif kwargs['pq_func'] == 'okamoto':
            self.func = Okamoto(**kwargs)  
        elif kwargs['pq_func'] == 'okamoto_evol':
            self.func = OkamotoEvolving(**kwargs)
        elif kwargs['pq_func'] in ['schechter', 'plexp']:
            self.func = Schechter(**kwargs)
        elif kwargs['pq_func'] in ['schechter_evol']:
            self.func = SchechterEvolving(**kwargs)
        else:
            raise NotImplemented('help')
            
    def __call__(self, **kwargs):
        
        # Patch up kwargs. Make sure inputs are arrays and that they lie
        # within the specified range (if there is one).
        kw = {}
        for key in kwargs:
            var = np.atleast_1d(kwargs[key])
            
            if key != self.func.x:
                kw[key] = var
                continue
            
            # Should have these options for var2 also    
            if self.func.var_ceil is not None:
                if type(self.func.var_ceil) in numeric_types:
                    var = np.minimum(var, self.func.var_ceil)
            if self.func.var_floor is not None:
                if type(self.func.var_floor) in numeric_types:
                    var = np.maximum(var, self.func.var_floor)
                
            kw[key] = var

        y = self.func.__call__(**kw)
        
        if self.func.val_ceil is not None:
            if type(self.func.val_ceil) in numeric_types:
                y = np.minimum(y, self.func.val_ceil)
        if self.func.val_floor is not None:
            if type(self.func.val_floor) in numeric_types:
                y = np.maximum(y, self.func.val_floor)
            
        return y
        
        
class ParameterizedQuantityOld(object):
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
            for i in range(10):
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

                #is_dim = None
                #if x.ndim == 2:
                #    d0 = np.diff(x, axis=0)
                #    if np.all(d0 == 0):
                #        
                #        # These are the unique x values
                #        if var == '1+z':
                #            xnew = x[0,:] - 1.
                #        else:    
                #            xnew = x[0,:]
                #        
                #        is_dim = 1
                #        
                #        var2 = 
                #        
                #        kw = {var: xnew, }
                #        
                #    elif np.all(d1 == 0):
                #        # These are the unique x values
                #        xnew = x[:,0]
                #        raise NotImplemented('help')
                #    else:
                #        kw = kwargs
                #else:    
                #    kw = kwargs
                
                _val = PQ.__call__(**kwargs)
                
                if type(_val) == np.ndarray:
                
                    # This is a memory-conserving trick. Remove redundant elements
                    # in input arrays before storing as attribute.
                    if _val.ndim == 2:
                        
                        d0 = np.all(np.diff(_val, axis=0) == 0)
                        d1 = np.all(np.diff(_val, axis=1) == 0)
                        if d0 and d1:
                            val = _val[0,0]
                        elif d0:
                            val = _val[0,:][None,:] 
                        elif d1:
                            val = _val[:,0][:,None]
                        else:
                            val = _val
                                
                        del _val
                            
                    else:
                        val = _val  
                else:
                    val = _val              
                
                
                #if is_dim == 1:
                #    val = _val[None,:]
                #elif is_dim is None:
                #    val = _val
                #else:
                #    raise NotImplemented('help')
                
                #print("HI HELLO", func, var, val.shape, x.shape, kwargs.keys())
                
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
            #print('{0} {1} {2} {3} {4}'.format(x, kwargs['z'], self.p0, self.p1, self.p2))
            f = self.p0 * (x / self.p1)**self.p2
        elif func == 'schechter':
            f = self.p0 * (x / self.p1)**self.p2 * np.exp(-(x / self.p1)) / self.p1
        elif func == 'schechter_mags':
            f = 0.4 * np.log(10.) * self.p0 \
                * (10**(0.4 * (self.p1 - x)))**(self.p2+1.) \
                * np.exp(-10**(0.4 * (self.p1 - x)))    
        # 'quadratic_lo' means higher order terms vanish when x << p3
        elif func == 'linear':
            f = self.p0 + self.p2 * (x - self.p1)
        elif func == 'loglinear':
            logf = self.p0 + self.p2 * (x - self.p1)
            f = 10**logf
        elif func == 'quadratic_lo':
            f = self.p0 * (1. +  self.p1 * (x / self.p3) + self.p2 * (x / self.p3)**2)
        # 'quadratic_hi' means higher order terms vanish when x >> p3
        elif func in ['quadratic_hi', 'quad']:
            f = self.p0 * (1. +  self.p1 * (self.p3 / x) + self.p2 * (self.p3 / x)**2)
        elif func == 'cubic_lo':
            f = self.p1 * (1. + self.p2 * (x / self.p0) + self.p3 * (x / self.p0)**2 + self.p4 * (x / self.p0)**3)
        elif func == 'cubic_hi':
            f = self.p1 * (1. +  self.p2 * (self.p0 / x) + self.p3 * (self.p0 / x)**2)
        elif func == 'poly':
            k = 0
            f = 0.0
            while 'p{}'.format(k) in self.__dict__:                
                norm = getattr(self, 'p{}'.format(2*k))
                slope = getattr(self, 'p{}'.format(2*k+1))
                
                if norm is None:
                    break
                
                f += norm * x**slope
                k += 1
                                            
        elif func == 'exp':
            f = self.p0 * np.exp((x / self.p1)**self.p2)
        elif func == 'exp-':
            f = self.p0 * np.exp(-(x / self.p1)**self.p2)    
        elif func == 'exp_flip':
            f = 1. - self.p0 * np.exp(-(x / self.p1)**self.p2)
        elif func == 'plexp':
            f = self.p0 * (x / self.p1)**self.p2 * np.exp(-(x / self.p3)**self.p4)
        elif func == 'dpl':
            xx = x / self.p1
            f = 2. * self.p0 / (xx**-self.p2 + xx**-self.p3)            
        elif func == 'dpl_arbnorm':
            xx = x / self.p1
            normcorr = (((self.p4 / self.p1)**-self.p2 + (self.p4 / self.p1)**-self.p3))
            f = self.p0 * normcorr / (xx**-self.p2 + xx**-self.p3)
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
            if type(self.floor) in [int, float, np.float64]:
                f = np.maximum(f, self.floor)
            else:
                f = np.maximum(f, self.floor(**kwargs))
                                          
        return f
              


