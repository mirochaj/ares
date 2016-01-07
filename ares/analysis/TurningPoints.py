"""

TurningPoints.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 10:14:06 MDT 2014

Description: 

"""

import numpy as np
from ..util import ParameterFile
from ..physics.Constants import nu_0_mhz
from ..util.Math import central_difference, take_derivative
from ..util.SetDefaultParameterValues import SetAllDefaults

try:
    from scipy.misc import derivative
    from scipy.optimize import minimize
    from scipy.interpolate import splrep, splev
except ImportError:
    pass

turning_points = list('ABCDE')

class TurningPoints(object):
    def __init__(self, inline=False, **kwargs):
        self.pf = ParameterFile(**kwargs)
            
        self.z_delay = self.pf['stop_delay']   
            
        # Keep track of turning points we've passed
        self.TPs = []
        #if self.pf['initial_redshift'] < 70:
        #    self.TPs.append('A')
            
        self.turning_points = {}    
        self.found_TP, self.z_TP = False, -99999
        self.Npts = 0    
        
    @property
    def step(self):
        if not hasattr(self, '_step'):
            self._step = 0
        
        return self._step

    def which_extremum(self, z, dTb, zp, dTbdz):
        # Compute second derivative
        zpp, dTb2dz2 = central_difference(zp, dTbdz)   
        
        negative = np.all(dTb < 0)
        positive = not negative
        concave_up = np.all(dTb2dz2 > 0)
        concave_down = not concave_up
        
        # Based on sign of brightness temperature and concavity,
        # determine which turning point we've found.
        if negative and concave_up and (max(z) > 60 and 'B' not in self.TPs):
            return 'A'
        elif negative and concave_down and ('B' not in self.TPs):
            return 'B'
        elif negative and concave_up and ('C' not in self.TPs) \
            and (max(z) < 60 or 'B' in self.TPs):
            return 'C'
        elif positive and concave_down and ('D' not in self.TPs):
            return 'D'
        else:
            return 'unknown'
        
    def is_stopping_point(self, z, dTb):
        """
        Determine if we have just passed a turning point.
        
        Parameters
        ----------
        z : np.ndarray
            Redshift (descending order).
        dTb : np.ndarray
            Corresponding brightness temperatures.
            
        """
        
        # Hack: don't check for turning points right at beginning
        if self.step < 10 or (z[-1] > 1e3):
            self._step += 1
            return False

        self._step += 1

        if not self.found_TP: 

            # Grab last three redshift and temperature points
            dz3pt = np.diff(z[-1:-4:-1])
            dT3pt = np.diff(dTb[-1:-4:-1])

            # If changes in temperatures have different signs, we found
            # a turning point. Come back in dz = self.z_delay to compute the 
            # details
            if len(np.unique(np.sign(dT3pt))) == 1:
                pass
            else:
                self.found_TP = True
                self.z_TP = z[-1] - self.z_delay
            
            # Check for zero-crossing too
            if not np.all(np.sign(dTb[-1:-3:-1]) < 0) and \
                'trans' not in self.turning_points:
                zTP = np.interp(0.0, dTb[-1:-3:-1], z[-1:-3:-1])
                zz, dTbdnu = take_derivative(np.array(z[-1:-10:-1]),
                    np.array(dTb[-1:-10:-1]), wrt='nu')

                slope = np.interp(zTP, zz, dTbdnu)
                
                self.turning_points['trans'] = (zTP, slope, None)
                
                if self.pf['stop'] == 'trans':
                    return True
        
        # If we found a turning point, hone in on its position
        if self.found_TP and (z[-1] < self.z_TP) and (self.Npts > 5): 
                        
            # Redshift and temperature points bracketing crudely-determined
            # extremum position
            k = np.argmin(np.abs(z - self.z_TP - 2 * self.z_delay))
            zbracketed = np.array(z[k:-1])
            Tbracketed = np.array(dTb[k:-1])
                        
            # Interpolate derivative to find extremum more precisely
            zz, dT = central_difference(zbracketed, Tbracketed)
            # In order of increasing redshift
                        
            TP = self.which_extremum(zbracketed, Tbracketed, zz, dT)
            self.TPs.append(TP)
            
            #if TP not in list('ABCDE'):
            #    raise ValueError('Unrecognized turning point!')
             
            # Crude guess at turning pt. position
            zTP_guess = zz[np.argmin(np.abs(dT))]    
            TTP_guess = np.interp(zTP_guess, z[k:-1], dTb[k:-1]) 
                                                                   
            # Spline interpolation to get "final" extremum redshift
            Bspl_fit1 = splrep(z[k:-1][-1::-1], dTb[k:-1][-1::-1], k=5)
                
            if TP in ['B', 'D']:
                dTb_fit = lambda zz: -splev(zz, Bspl_fit1)
            else:
                dTb_fit = lambda zz: splev(zz, Bspl_fit1)
            
            zTP = minimize(dTb_fit, zTP_guess, tol=1e-4).x[0]
            TTP = dTb_fit(zTP)
                            
            if TP in ['B', 'D']:
                TTP *= -1.
                            
            # Compute curvature at turning point (mK**2 / MHz**2)
            nuTP = nu_0_mhz / (1. + zTP)
            d2 = derivative(lambda zz: splev(zz, Bspl_fit1), x0=float(zTP), 
                n=2, dx=1e-4, order=5) * nu_0_mhz**2 / nuTP**4
                
            self.turning_points[TP] = (float(zTP), float(TTP), float(d2))
                      
            self.found_TP = False
            self.z_TP = -99999
            
            self.Npts = 0
                              
            if self.pf['stop'] in self.TPs:
                return True
        
        elif self.found_TP:
            self.Npts += 1
            
        return False
            