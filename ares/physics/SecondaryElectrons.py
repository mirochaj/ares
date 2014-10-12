"""
SecondaryElectrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-11-07.

Description: Read in Furlanetto & Stoever results, provide functions for 
interpolation of heating and ionization deposition fractions for fast 
secondary electrons.  Fits of Shull & vanSteenberg (1985) and Ricotti, 
Gnedin, & Shull (2002) also available.

"""

import os
import numpy as np
from collections import Iterable

try:
    from mathutils.interpolate import LinearNDInterpolator
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass

# If anything is identically zero for methods 2 and 3, 
# our spline will get screwed up since log(0) = inf
tiny_number = 1e-20

class SecondaryElectrons:
    def __init__(self, method=0):
        self.Method = method

        if self.Method == 3:
            rt1d = os.environ.get("RT1D")
            if rt1d:
                self.fn = "%s/input/secondary_electron_data.hdf5" % rt1d
                f = h5py.File(self.fn, 'r')
            else:
                raise Exception('Error loading secondary electron data.')    
                
            # Read in Furlanetto & Stoever lookup tables
            self.E = f["electron_energy"].value
            self._x = f["ionized_fraction"].value
            self._logx = np.log10(self.x)
            
            self.fh_tab = f["f_heat"].value
            self.fionHI_tab = f["fion_HI"].value
            self.fionHeI_tab = f["fion_HeI"].value
            self.fionHeII_tab = f["fion_HeII"].value
            self.fexc_tab = f["fexc"].value
            self.flya_tab = f['f_Lya'].value
            
            from scipy.interpolate import RectBivariateSpline, interp2d
            
            self.fh = RectBivariateSpline(self.E, self.x, f["f_heat"].value)
            self.fHI = RectBivariateSpline(self.E, self.x, f["fion_HI"].value)
            self.fHeI = RectBivariateSpline(self.E, self.x, f["fion_HeI"].value)
            self.fHeII = RectBivariateSpline(self.E, self.x, f["fion_HeII"].value)
            self.fexc = interp2d(self.E, self.x, f["fexc"].value, kind='linear')
            self.flya = RectBivariateSpline(self.E, self.x, f['f_Lya'].value)
            
            f.close()
            
    @property
    def logx(self):
        if not hasattr(self, '_logx'):
            self._logx = np.arange(-4, 0.1, 0.1)
        return self._logx
    
    @property
    def x(self):
        if not hasattr(self, '_x'):
            self._x = 10**self.logx
        return self._x    
        
    def DepositionFraction(self, xHII, E=None, channel='heat'):
        """
        Return the fraction of secondary electron energy deposited as heat, or 
        further ionizations.
        
        The parameter 'channel' determines which we want, and could be:
        
            channel = (heat, h_1, he_1, he_2, lya)
        
        also,
                    
            Method = 0: OFF - all secondary electron energy goes to heat.
            Method = 1: Empirical fits of Shull & vanSteenberg 1985.
            Method = 2: Empirical Fits of Ricotti et al. 2002.
            Method = 3: Lookup tables of Furlanetto & Stoever 2010.
            
        xHII is preferably an array of values (corresponding to grid elements).
            
        """
        
        if not isinstance(xHII, Iterable):
            xHII = np.array([xHII])
                    
        if E is None: 
            E = tiny_number
        
        if self.Method == 0:
            if channel == 'heat':
                return np.ones_like(xHII)
            else: 
                return np.zeros_like(xHII)
            
        if self.Method == 1: 
            if channel == 'heat': 
                tmp = tiny_number * np.zeros_like(xHII)
                tmp[xHII <= 1e-4] = 0.15 * np.ones(len(tmp[xHII <= 1e-4]))
                tmp[xHII > 1e-4] = 0.9971 * (1. - pow(1 - 
                    pow(xHII[xHII > 1e-4], 0.2663), 1.3163))
                return tmp
            if channel == 'h_1': 
                return 0.3908 * pow(1. - pow(xHII, 0.4092), 1.7592)
            if channel == 'he_1': 
                return 0.0554 * pow(1. - pow(xHII, 0.4614), 1.6660) 
            if channel == 'he_2': 
                return tiny_number * np.zeros_like(xHII)
            if channel == 'lya': # Assuming that ALL excitations lead to a LyA photon
                return 0.4766 * pow(1. - pow(xHII, 0.2735), 1.5221)
            
        # Ricotti, Gnedin, & Shull (2002)
        if self.Method == 2:
            if channel == 'heat': 
                tmp = tiny_number * np.zeros_like(xHII)
                tmp[xHII <= 1e-4] = 0.15 * np.ones_like(tmp[xHII <= 1e-4]) 
                if E >= 11:
                    tmp[xHII > 1e-4] = 3.9811 * (11. / E)**0.7 \
                        * pow(xHII[xHII > 1e-4], 0.4) * \
                        (1. - pow(xHII[xHII > 1e-4] , 0.34))**2 + \
                        (1. - (1. - pow(xHII[xHII > 1e-4] , 0.2663))**1.3163)
                else:
                    tmp[xHII > 1e-4] = (1. - tiny_number) \
                        * np.ones_like(tmp[xHII <= 1e-4]) 
                    
                return tmp
                    
            if channel == 'h_1': 
                if E >= 28:
                    return np.maximum(-0.6941 * (28. / E)**0.4 * pow(xHII, 0.2) * \
                        (1. - pow(xHII, 0.38))**2 + \
                        0.3908 * (1. - pow(xHII, 0.4092))**1.7592, tiny_number)
                else:
                    return tiny_number * np.zeros_like(xHII)
            if channel == 'he_1': 
                if E >= 28:
                    return np.maximum(-0.0984 * (28. / E)**0.4 * pow(xHII, 0.2) * \
                        (1. - pow(xHII, 0.38))**2 + \
                        0.0554 * (1. - pow(xHII, 0.4614))**1.6660, tiny_number)
                else:
                    return tiny_number * np.zeros_like(xHII)
            if channel == 'he_2': 
                return tiny_number * np.zeros_like(xHII)
        
        # Furlanetto & Stoever (2010)
        if self.Method == 3:
            
            f = tiny_number * np.zeros_like(xHII)
            
            for i, x in enumerate(xHII):
            
                if channel == 'heat':
                    f[i] = self.fh(E, x)
                if channel == 'h_1': 
                    f[i] = self.fHI(E, x)
                if channel == 'he_1': 
                    f[i] = self.fHeI(E, x)
                if channel == 'he_2': 
                    f[i] = self.fHeII(E, x)
                if channel == 'lya':
                    f[i] = self.flya(E, x)
                if channel == 'exc':
                    f[i] = self.fexc(E, x)
            
            return f
            
