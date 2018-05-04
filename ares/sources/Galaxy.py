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
from scipy.interpolate import RectBivariateSpline

try:
    from scipy.interpolate import Akima1DInterpolator, interp1d
except ImportError:
    pass

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
                self._tab_sed_ = self.src.data  
        return self._tab_sed_

    @property
    def _sfr_func(self):
        if not hasattr(self, '_sfr_func_'):
            if self.pf['source_sfh'] is None:
                self._sfr_func_ = lambda t: np.ones_like(t)
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
            # i.e., not in lookup table. Must do this to force Synth model
            # to load all metallicities.
            tmp['source_Z'] = 0.0245 
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
                    self._synth_model._data_all_Z[:,i_lam,:])
                self._synth_interp = interp2d
                
                sed = []
                for t in tarr:
                    sed.append(float(interp2d(self.Z(t), t)))
                
                sed = np.array(sed)
            else:
                sed = self._tab_sed[i_lam]
                
            #interp_logt = Akima1DInterpolator(np.log(self.src.times), 
            #    np.log(sed))    
            interp_logt = interp1d(np.log(self.src.times),     
                np.log(sed), bounds_error=False, 
                fill_value=(0., np.log(sed)[-1]))

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
        
        if dt < 1:
            print("dt < 1 might cause problems.")
                
        # Setup interpolant for simple stellar population
        interp_t = self.create_interpolant(times, band=band, 
            wavelength=wavelength)
        
        # Must change if band is not None
        Nt = len(times)
        sed = np.zeros(Nt)
        
        # Youngest stellar population we've got.
        t0 = min(times[times > 0])
        
        # Construct SED one time at a...time.
        sfr = self.SFR(times)
        L = interp_t(times)
        
        # New stars only
        sed += sfr * L[0] * dt
                
        # If aging is being treated, must sum up luminosity from all
        # past 'episodes' of star formation.
        if self.src.pf['source_aging']:
            for j in range(1, Nt):
                sed[j] += np.sum(L[0:j][-1::-1] * sfr[0:j] * dt)
        
        self.sed = sed
        return sed
