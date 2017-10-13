"""

Galaxy.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jun  8 13:01:27 PDT 2017

Description: 

"""

import numpy as np
from ..util import ParameterFile
from .SynthesisModel import SynthesisModel
from scipy.interpolate import Akima1DInterpolator, RectBivariateSpline

class Galaxy(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.pf = ParameterFile(**kwargs)

    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = SynthesisModel(**self.pf)

        return self._src

    @property
    def _tab_E(self):
        if not hasattr(self, '_tab_E_'):
            if self.src.pf['source_degradation'] > 0:
                raise NotImplemented('help?')
            else:
                self._tab_E_ = self.src.energies
                
    @property
    def _tab_sed(self):
        if not hasattr(self, '_tab_sed_'):
            if self.src.pf['source_degradation'] > 0:
                raise NotImplemented('help?')
            else:
                # Tables are for a 10^6 Msun burst
                self._tab_sed_ = self.src.data / 1e6 
        return self._tab_sed_

    @property
    def _sfr_func(self):
        if not hasattr(self, '_sfr_func_'):
            if self.pf['source_sfh'] is None:
                self._sfr_func_ = lambda t: 1.0
                print("Defaulting to 1 Msun/yr")
            else:
                self._sfr_func_ = self.pf['source_sfh']
                
        return self._sfr_func_

    def SFR(self, t):
        return self._sfr_func(t)
    
    @property
    def _synth_model(self):
        if not hasattr(self, '_synth_model_'):
            tmp = self.kwargs
            tmp['source_Z'] = 0.0245 # i.e., not in lookup table
            self._synth_model_ = SynthesisModel(**tmp)
            junk = self._synth_model.L1600_per_sfr
            #data_all_Z = pop._data_all_Z
        return self._synth_model_
    
    @property
    def _Z_func(self):
        if not hasattr(self, '_Z_func_'):
            if self.pf['source_meh'] is None:
                self._Z_func_ = lambda t: self.pf['source_Z']
                print("Defaulting to constant Z={0:f}".format(\
                    self.pf['source_Z']))
            else:
                self._Z_func_ = self.pf['source_meh']
    
        return self._Z_func_
        
    def Z(self, t):
        return self._Z_func(t)
        
    def create_interpolant(self, tout, wavelength=None, band=None):
        
        assert wavelength is not None or band is not None
        
        if wavelength is not None:
            
            i_lam = np.argmin(np.abs(wavelength - self.src.wavelengths))
            
            if self.pf['source_meh'] is not None:                
                Zarr = np.sort(list(self._synth_model.metallicities.values()))
                tarr = self._synth_model.times
                
                interp2d = RectBivariateSpline(Zarr, tarr, 
                    self._synth_model._data_all_Z[:,i_lam,:] / 1e6)
                
                sed = []
                for t in tarr:
                    sed.append(float(interp2d(self.Z(t), t)))
                
                sed = np.array(sed)
                
            else:
                sed = self._tab_sed[i_lam]
                
            interp_logt = Akima1DInterpolator(np.log(self.src.times), 
                np.log(sed))    
                
                
            interp_t = lambda t: np.exp(interp_logt.__call__(np.log(t)))
            
            return interp_t
            
        elif band is not None:
            sed = np.zeros((self._tab_E.size, len(times)))
            raise NotImplemented('ehlp')

    def generate_history(self, times, band=None, wavelength=None):
        """
        For a given star formation history, compute the spectrum.
        """
        
        assert self.src.pf['source_ssp']
        
        dt = np.diff(times)
        assert np.allclose(dt, np.roll(dt, 1)), \
            "Time points must be evenly spaced!"
        dt = dt[0] * 1e6
                
        # Setup interpolant for simple stellar population
        interp_t = self.create_interpolant(times, band=band, 
            wavelength=wavelength)
        
        # Must change if band is not None
        sed = np.zeros(len(times))
        
        # Youngest stellar population we've got.
        t0 = min(times[times > 0])
        
        # Loop over time
        for i_t1, t1 in enumerate(times):

            if t1 == 0:
                continue

            # Light from new stars
            sed[i_t1] += interp_t(t0) * self.SFR(t1) * dt 

            if not self.src.pf['source_aging']:
                continue

            # On each timestep, need to include new emission and 
            # the evolved emission from previous star formation

            if wavelength is not None:
                
                # Loop over the past
                for i_t2, t2 in enumerate(times):
                    if t2 == 0:
                        continue
                    if t2 >= t1:
                        break

                    # Light from old stars. Grab the SFR from the appropriate
                    # moment in the past, multiply by dt (to get mass formed),
                    # and add flux from those stars aged by t1-t2
                    sed[i_t1] += interp_t(t1-t2) * self.SFR(t2) * dt

            elif band is not None:
                raise NotImplemented('help')
                
                
            
            
        
        self.sed = sed
        return sed
