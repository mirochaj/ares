"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import numpy as np
from ..util import ParameterFile    
from .MultiPhaseIGM import MultiPhaseIGM

class Global21cm:
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.
        
        ..note :: This is essentially a MultiPhaseIGM calculation, except
            with the option of performing inline analysis to track extrema 
            in the 21-cm signal, and/or use alternative (phenomenological)
            parameterizations such as a tanh.
            
        """

        # See if this is a tanh model calculation
        is_tanh = self._check_if_tanh(**kwargs)

        if not is_tanh:
            self.pf = ParameterFile(**kwargs)
        else:
            return

        # If a physical model, proceed with initialization
        self.igm = MultiPhaseIGM(**self.pf)

        # Inline tracking of turning points
        if self.pf['track_extrema']:
            from ..analysis.TurningPoints import TurningPoints
            self.track = TurningPoints(inline=True, **self.pf)    
        
    def _check_if_tanh(self, **kwargs):
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
    
        return True
        
    def run(self):
        """
        Run a 21-cm simulation.
        """
        
        # do stuff here
        
        #if self.pf['track_extrema']:
        #    stop = self.track.is_stopping_point(self.history['z'],
        #        self.history['dTb'])
        #    if stop:
        #        break
    
        
    def step(self):
        """
        Generator for the 21-cm signal.
        """