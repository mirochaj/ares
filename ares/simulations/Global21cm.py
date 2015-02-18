"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import numpy as np
from ..util import ParameterFile    
from .TwoZoneIGM import TwoZoneIGM

class Global21cm:
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.

        See Also
        --------
        Set of all acceptable kwargs in:
            ares/util/SetDefaultParameterValues.py
            
        """
        
        # See if this is a tanh model calculation
        is_tanh = self._check_if_tanh(**kwargs)
        
        if not is_tanh:
            self.pf = ParameterFile(**kwargs)
        else:
            return
            
        # If a physical model, proceed with initialization
        self.igm = TwoZoneIGM(**self.pf)

    def _check_if_tanh(self, kwargs):
        if not kwargs:
            return False
    
        if 'tanh_model' not in kwargs:
            return False

        if not kwargs['tanh_model']:
            return False

        from ..util.TanhModel import TanhModel

        tanh_model = TanhModel(**kwargs)
        self.pf = tanh_model.pf

        if self.pf['tanh_nu'] is not None:
            nu = self.pf['tanh_nu']
            z = nu_0_mhz / nu - 1.
        else:
            z = np.arange(self.pf['final_redshift'] + self.pf['tanh_dz'],
                self.pf['initial_redshift'], self.pf['tanh_dz'])[-1::-1]

        self.history = tanh_model(z, **self.pf).data

        self.grid = Grid(dims=1)
        self.grid.set_cosmology(
            initial_redshift=self.pf['initial_redshift'],
            omega_m_0=self.pf["omega_m_0"],
            omega_l_0=self.pf["omega_l_0"],
            omega_b_0=self.pf["omega_b_0"],
            hubble_0=self.pf["hubble_0"],
            helium_by_number=self.pf['helium_by_number'], 
            cmb_temp_0=self.pf["cmb_temp_0"],
            approx_highz=self.pf["approx_highz"])
    
        return True
    
        
