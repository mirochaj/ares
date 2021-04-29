"""

DustCorrection.py

Author: Jason Sun and Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 17:55:27 PST 2016

Description: 

"""

import numpy as np
from types import FunctionType
from ..util import ParameterFile
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

_coeff = \
{
 'meurer1999': [4.43, 1.99],
 'pettini1998': [1.49, 0.68],
 'capak2015': [0.312, 0.176],
}

_magarr = np.arange(-35, 0, 0.1)
          
class DustCorrection(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    @property
    def method(self):
        if not hasattr(self, '_method'):
            if self.pf['dustcorr_method'] is None:
                self._method = None
            elif type(self.pf['dustcorr_method']) is list:
                meth = self.pf['dustcorr_method']
                self._method = \
                    [meth[i] for i in range(len(meth))]
                    
                assert len(self._method) == len(self.pf['dustcorr_ztrans'])
            else:
                self._method = self.pf['dustcorr_method']

        return self._method
    
    #   ==========   Parametrization of Auv   ==========   #
    
    def AUV(self, z, mag):
        """
        Return non-negative mean correction <Auv> averaged over Luv assuming a
        normally distributed Auv
        """
        
        
        ##
        # Physical models!
        ##
        if self.method == 'physical':
            raise NotImplemented('help')
        
        ##
        # Empirical prescriptions.
        ##
        
        
        if self.method is None:
            method = None
        elif type(self.method) is list:
            for i, method in enumerate(self.method):
                if z < self.pf['dustcorr_ztrans'][i]:
                    continue
                if i == (len(self.method) - 1):
                    break    
                if z <= self.pf['dustcorr_ztrans'][i+1]:
                    break
        else:
            method = self.method
        if method is None:
            return 0.0
        a, b = _coeff[method]
        beta = self.Beta(z, mag)
        s_a = self.pf['dustcorr_scatter_A']
        s_b = self.pf['dustcorr_scatter_B']
        sigma = np.sqrt(s_a**2 + (s_b * beta)**2)
        AUV = a + b * beta + 0.2 * np.log(10) * sigma
        return np.maximum(AUV, 0.0)
        
    def _Mobs_func(self, z):
        if not hasattr(self, '_Mobs_func_'):
            self._Mobs_func_ = {}
            
        if z not in self._Mobs_func_:
            y = []
            for M in _magarr:
                f_AUV = lambda mag: self.AUV(z, mag)
                
                to_min = lambda xx: np.abs(xx - f_AUV(xx) - M)
                y.append(fsolve(to_min, M)[0])
                
            y = np.array(y)
            
            self._Mobs_func_[z] = lambda M: np.interp(M, _magarr, y,
                left=np.nan, right=0.0)
            
        return self._Mobs_func_[z]    
        
    def Mobs(self, z, MUV):
        """
        Return observed  magnitude.
        """
        
        ##
        # Physical models!
        ##
        if self.method == 'physical':
            raise NotImplemented('help')
        
        # Compute the absolute magnitude one measured in the first place,
        # i.e., before correcting for dust.
        
        func = self._Mobs_func(z)
        
        Mobs = func(MUV)
        
        return Mobs
        
           
        #if type(MUV) in [np.ndarray, np.ma.core.MaskedArray, list, tuple]:
        #    
        #    x = []
        #    for M in MUV:
        #        f_AUV = lambda mag: self.AUV(z, mag)
        #        
        #        to_min = lambda xx: np.abs(xx - f_AUV(xx) - M)
        #        x.append(fsolve(to_min, M)[0])
        #        
        #    x = np.array(x)    
        #else:
        #
        #    f_AUV = lambda mag: self.AUV(z, mag)
        #    
        #    to_min = lambda xx: np.abs(xx - f_AUV(xx) - MUV)
        #    x = fsolve(to_min, MUV)[0]
        #
        #return x

    #   ==========   Parametrization of Beta   ==========   #
    def Beta(self, z, mag):
        
        ##
        # Physical models!
        ##
        if self.method == 'physical':
            raise NotImplemented('help')
            
            # Need a population instance.
        
        if isinstance(self.pf['dustcorr_beta'], basestring):
            return self._beta_fit(z, mag)
        elif type(self.pf['dustcorr_beta']) is FunctionType:    
            return self.pf['dustcorr_beta'](z, mag)
        else:
            return self.pf['dustcorr_beta'] * np.ones_like(mag)
            
    def _bouwens2014_beta0(self, z):
        """
        Get the measured UV continuum slope from Bouwens+2014 (Table 3).
        """
        
        _z = round(z,0)
        
        if _z < 4.0:
            val = -1.70; err_rand = 0.07; err_sys = 0.15
        elif _z == 4.0:
            val = -1.85; err_rand = 0.01; err_sys = 0.06
        elif _z == 5.0:
            val = -1.91; err_rand = 0.02; err_sys = 0.06
        elif _z == 6.0:
            val = -2.00; err_rand = 0.05; err_sys = 0.08
        elif _z == 7.0:
            val = -2.05; err_rand = 0.09; err_sys = 0.13
        elif _z == 8.0:
            val = -2.13; err_rand = 0.44; err_sys = 0.27
        else:
            # Assume constant at z>8
            val = -2.00; err_rand = 0.00; err_sys = 0.00
                    
        return val, err_rand, err_sys
        
    def _bouwens2014_dbeta0_dM0(self, z):
        """
        Get the measured slope of the UV continuum slope from Bouwens+2014.
        """
        
        _z = round(z,0)
        if _z < 4.0:
            val = -0.2; err = 0.04
        elif _z == 4.0:
            val = -0.11; err = 0.01
        elif _z == 5.0:
            val = -0.14; err = 0.01
        elif _z == 6.0:
            val = -0.20; err = 0.02
        elif _z == 7.0:
            val = -0.20; err = 0.04
        elif _z == 8.0:
            val = -0.20; err = 0.07
        else:
            # Assume constant at z>8
            val = -0.15; err = 0.00
                        
        return val, err
    
    def _beta_fit(self, z, mag):
        """
        An linear + exponential fit to Bouwens+14 data adopted from Mason+2015.
        """
        if self.pf['dustcorr_beta'] == 'bouwens2012':
            # This is mentioned in the caption of Smit et al. 2012 Fig 1
            dbeta_dMUV = -0.11
            return dbeta_dMUV * (mag + 19.5) - 2.00
        elif self.pf['dustcorr_beta'] == 'bouwens2014':
            # His Table 3
            beta0 = self._bouwens2014_beta0(z)[0]
            dbeta_dMUV = self._bouwens2014_dbeta0_dM0(z)[0]
            return dbeta_dMUV * (mag + 19.5) + beta0    
        elif self.pf['dustcorr_beta'] == 'mason2015':
            _M0 = -19.5; _c = -2.33
            # Must handle piecewise function carefully for arrays of magnitudes
            # lo vs. hi NOT in absolute value, i.e., lo means bright.
            if type(mag) == np.ndarray:
                assert np.all(np.diff(mag) > 0), \
                    "Magnitude values must be increasing!"
                Mlo = mag[mag < _M0]
                Mhi = mag[mag >= _M0]
                Alo = self._bouwens2014_dbeta0_dM0(z)[0]*(Mlo - _M0) + self._bouwens2014_beta0(z)[0]
                Ahi = (self._bouwens2014_beta0(z)[0] - _c) * np.exp(self._bouwens2014_dbeta0_dM0(z)[0]*(Mhi - _M0)/(self._bouwens2014_beta0(z)[0] - _c)) + _c
                return np.concatenate((Alo, Ahi))
            # Otherwise, standard if/else works
            if mag < _M0:
                return self._bouwens2014_dbeta0_dM0(z)[0]*(mag - _M0) + self._bouwens2014_beta0(z)[0]
            else:
                return (self._bouwens2014_beta0(z)[0] - _c) * np.exp(self._bouwens2014_dbeta0_dM0(z)[0]*(mag - _M0)/(self._bouwens2014_beta0(z)[0] - _c)) + _c
        else:
            raise NotImplementedError('Unrecognized dustcorr: {!s}'.format(\
                self.pf['dustcorr_beta']))

