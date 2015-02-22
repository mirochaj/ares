"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import numpy as np
from ..util.ReadData import _sort_history
from ..util import ParameterFile, ProgressBar
from .MultiPhaseMedium import MultiPhaseMedium

class Global21cm:
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.
        
        ..note :: This is essentially a MultiPhaseMedium calculation, except
            with the option of performing inline analysis to track extrema 
            in the 21-cm signal, and/or use alternative (phenomenological)
            parameterizations such as a tanh for the ionization, thermal,
            and LW background evolution.
            
        """

        # See if this is a tanh model calculation
        is_tanh = self._check_if_tanh(**kwargs)

        if not is_tanh:
            self.pf = ParameterFile(**kwargs)
        else:
            return

        # If a physical model, proceed with initialization
        self.medium = MultiPhaseMedium(**self.pf)

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
        
        # If this was a tanh model, we're already dones.
        if hasattr(self, 'history'):
            return
        
        tf = self.medium.tf
                
        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        pb.start()
        
        # List for extrema-finding    
        all_dTb = []
        
        # Lists for data in general
        all_t = []; all_z = []
        all_data_igm = []; all_data_cgm = []
        for t, z, data_igm, data_cgm in self.step():
            
            pb.update(t)
            
            # Save data
            all_z.append(z)
            all_t.append(t)
            all_dTb.append(data_igm['dTb'][0])
            all_data_igm.append(data_igm.copy()) 
            all_data_cgm.append(data_cgm.copy())
            
            # Automatically find turning points
            if self.pf['track_extrema']:
                if self.track.is_stopping_point(all_z, all_dTb):
                    break
            
        pb.finish()

        self.history_igm = _sort_history(all_data_igm, prefix='igm_',
            squeeze=True)
        self.history_cgm = _sort_history(all_data_cgm, prefix='cgm_',
            squeeze=True)
        
        self.history = self.history_igm.copy()
        self.history.update(self.history_cgm)
        self.history['t'] = np.array(all_t)
        self.history['z'] = np.array(all_z)
        
    def step(self):
        """
        Generator for the 21-cm signal.
        
        .. note:: Basically just calling MultiPhaseMedium here, except we
            compute the spin temperature and brightness temperature.
        
        Returns
        -------
        Generator for MultiPhaseIGM object, with notable addition that
        the spin temperature and 21-cm brightness temperature are now 
        tracked.

        """

        
        for t, z, data_igm, data_cgm in self.medium.step():            
                        
            # Grab Lyman alpha flux
            Ja = self.medium.field.LymanAlphaFlux(z)
            #fluxes = 
            
            # Compute spin temperature
            n_H = self.medium.parcel_igm.grid.cosm.nH(z)
            Ts = \
                self.medium.parcel_igm.grid.hydr.Ts(
                    z, data_igm['Tk'], Ja, data_igm['h_2'], 
                    data_igm['e'] * n_H)

            # Derive brightness temperature
            dTb = \
                self.medium.parcel_igm.grid.hydr.dTb(
                    z, data_cgm['h_2'], Ts)
            
            # Add derived fields to data
            data_igm.update({'Ts': Ts, 'dTb': dTb})
                        
            # Yield!            
            yield t, z, data_igm, data_cgm
            
        
        