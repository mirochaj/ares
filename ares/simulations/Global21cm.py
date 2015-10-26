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
from ..physics.Constants import nu_0_mhz, E_LyA

defaults = \
{
 'load_ics': True,
 'problem_type': 101,
}

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
        is_phenom = self._check_if_phenom(**kwargs)

        if is_phenom:
            return
        
        kwargs.update(defaults)
        self.pf = ParameterFile(**kwargs)
            
        if not (self.pf['include_igm'] and self.pf['include_cgm']):
            raise ValueError('Can only compute 21-cm signal in two-phase medium!')
            
        # If a physical model, proceed with initialization
        self.medium = MultiPhaseMedium(**kwargs)

        # Inline tracking of turning points
        if self.pf['track_extrema']:
            from ..analysis.TurningPoints import TurningPoints
            self.track = TurningPoints(inline=True, **self.pf)
        
    @property
    def pops(self):
        return self.medium.field.pops
        
    @property
    def grid(self):
        return self.medium.field.grid
        
    def _init_dTb(self):
        """
        Compute differential brightness temperature for initial conditions.
        """
        z = self.all_z
        
        dTb = []
        for i, data_igm in enumerate(self.all_data_igm):
            
            n_H = self.medium.parcel_igm.grid.cosm.nH(z[i])
            Ts = \
                self.medium.parcel_igm.grid.hydr.Ts(
                    z[i], data_igm['Tk'], 0.0, data_igm['h_2'],
                    data_igm['e'] * n_H)
            
            # Compute volume-averaged ionized fraction
            QHII = self.all_data_cgm[i]['h_2']
            xavg = QHII + (1. - QHII) * data_igm['h_2']        
            
            # Derive brightness temperature
            Tb = self.medium.parcel_igm.grid.hydr.dTb(z[i], xavg, Ts)
            self.all_data_igm[i]['dTb'] = Tb
            self.all_data_igm[i]['Ts'] = Ts
            dTb.append(Tb)
            
        return dTb
        
    def _check_if_phenom(self, **kwargs):
        if not kwargs:
            return False
    
        if ('tanh_model' not in kwargs) and ('gaussian_model' not in kwargs):
            return False
            
        is_tanh = False
        is_gauss = False    

        if 'tanh_model' in kwargs:
            if kwargs['tanh_model']:
                from ..util.TanhModel import TanhModel as PhenomModel
                is_tanh = True
                
        elif 'gaussian_model' in kwargs:
            if kwargs['gaussian_model']:
                from ..util.GaussianSignal import GaussianModel as PhenomModel            
                is_gauss = True
                
        model = PhenomModel(**kwargs)                
        self.pf = model.pf
            
        if self.pf['output_frequencies'] is not None:
            nu = self.pf['output_frequencies']
            z = nu_0_mhz / nu - 1.
        else:
            z = np.arange(self.pf['final_redshift'] + self.pf['output_dz'],
                self.pf['initial_redshift'], self.pf['output_dz'])[-1::-1]
            nu =  nu_0_mhz / (1. + z)   
        
        if is_gauss:
            self.history = model(nu, **self.pf).data    
        else:
            self.history = model(z, **self.pf).data    

        return True
        
    def run(self):
        """
        Run a 21-cm simulation.
        
        Returns
        -------
        Nothing: sets `history` attribute.
        
        """
        
        # If this was a tanh model, we're already done.
        if hasattr(self, 'history'):
            return
        
        tf = self.medium.tf
                
        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        pb.start()
        
        # Lists for data in general
        self.all_t, self.all_z, self.all_data_igm, self.all_data_cgm, \
            self.all_RC_igm, self.all_RC_cgm = \
            self.medium.all_t, self.medium.all_z, self.medium.all_data_igm, \
            self.medium.all_data_cgm, self.medium.all_RCs_igm, self.medium.all_RCs_cgm
        
        # Add zeros for Ja
        for element in self.all_data_igm:
            element['Ja'] = 0.0
        
        # List for extrema-finding    
        self.all_dTb = self._init_dTb()
                                
        for t, z, data_igm, data_cgm, rc_igm, rc_cgm in self.step():
                        
            pb.update(t)
                    
            # Save data
            self.all_z.append(z)
            self.all_t.append(t)
            self.all_dTb.append(data_igm['dTb'][0])
            self.all_data_igm.append(data_igm.copy()) 
            self.all_data_cgm.append(data_cgm.copy())
            self.all_RC_igm.append(rc_igm.copy()) 
            self.all_RC_cgm.append(rc_cgm.copy())
            
            # Automatically find turning points
            if self.pf['track_extrema']:
                if self.track.is_stopping_point(self.all_z, self.all_dTb):
                    break

        pb.finish()

        self.history_igm = _sort_history(self.all_data_igm, prefix='igm_',
            squeeze=True)
        self.history_cgm = _sort_history(self.all_data_cgm, prefix='cgm_',
            squeeze=True)

        self.history = self.history_igm.copy()
        self.history.update(self.history_cgm)
        
        # Save rate coefficients [optional]
        if self.pf['save_rate_coefficients']:
            self.rates_igm = \
                _sort_history(self.all_RC_igm, prefix='igm_', squeeze=True)
            self.rates_cgm = \
                _sort_history(self.all_RC_cgm, prefix='cgm_', squeeze=True)
        
            self.history.update(self.rates_igm)
            self.history.update(self.rates_cgm)

        self.history['t'] = np.array(self.all_t)
        self.history['z'] = np.array(self.all_z)
        
        if self.pf['track_extrema']:
            self.run_inline_analysis()

    def step(self):
        """
        Generator for the 21-cm signal.
        
        .. note:: Basically just calling MultiPhaseMedium here, except we
            compute the spin temperature and brightness temperature on
            each step.
        
        Returns
        -------
        Generator for MultiPhaseMedium object, with notable addition that
        the spin temperature and 21-cm brightness temperature are now 
        tracked.

        """
                        
        for t, z, data_igm, data_cgm, RC_igm, RC_cgm in self.medium.step():            
                                                                                       
            # Grab Lyman alpha flux
            Ja = 0.0
            for i, pop in enumerate(self.medium.field.pops):
                if not pop.is_lya_src:
                    continue
                
                if not self.medium.field.solve_rte[i]:    
                    Ja += self.medium.field.LymanAlphaFlux(z)
                    continue

                # Grab line fluxes for this population for this step
                for j, band in enumerate(self.medium.field.bands_by_pop[i]):
                    E0, E1 = band
                    if not (E0 <= E_LyA < E1):
                        continue
                    
                    Earr = np.concatenate(self.medium.field.energies[i][j])
                    l = np.argmin(np.abs(Earr - E_LyA))    
                    
                    Ja += self.medium.field.all_fluxes[-1][i][j][l]
                                                                    
            # Compute spin temperature
            n_H = self.medium.parcel_igm.grid.cosm.nH(z)
            Ts = self.medium.parcel_igm.grid.hydr.Ts(z,
                data_igm['Tk'], Ja, data_igm['h_2'], data_igm['e'] * n_H)

            # Compute volume-averaged ionized fraction
            xavg = data_cgm['h_2'] + (1. - data_cgm['h_2']) * data_igm['h_2']

            # Derive brightness temperature
            dTb = self.medium.parcel_igm.grid.hydr.dTb(z, xavg, Ts)

            # Add derived fields to data
            data_igm.update({'Ts': Ts, 'dTb': dTb, 'Ja': Ja})

            # Yield!            
            yield t, z, data_igm, data_cgm, RC_igm, RC_cgm 

    def run_inline_analysis(self):    

        if (self.pf['inline_analysis'] is None) and \
           (self.pf['auto_generate_blobs'] == False):
            return

        tmp = {}
        for key in self.history:
            if type(self.history[key]) is list:
                tmp[key] = np.array(self.history[key])
            else:
                tmp[key] = self.history[key]

        self.history = tmp
    
        from ..analysis.InlineAnalysis import InlineAnalysis
        anl = InlineAnalysis(self)
        anl.run_inline_analysis()
    
        self.turning_points = anl.turning_points
    
        self.blobs = anl.blobs
        self.blob_names, self.blob_redshifts = \
            anl.blob_names, anl.blob_redshifts
            
    def save(self, prefix, suffix='pkl', clobber=False):
        """
        Save results of calculation. Pickle parameter file dict.
    
        Notes
        -----
        1) will save files as prefix.history.suffix and prefix.parameters.pkl.
        2) ASCII files will fail if simulation had multiple populations.
    
        Parameters
        ----------
        prefix : str
            Prefix of save filename
        suffix : str
            Suffix of save filename. Can be hdf5 (or h5), pkl, or npz. 
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?
    
        """
    
        import os
    
        fn = '%s.history.%s' % (prefix, suffix)
    
        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        if suffix == 'pkl':      
            import pickle
                  
            f = open(fn, 'wb')
            pickle.dump(self.history, f)
            f.close()
    
        elif suffix in ['hdf5', 'h5']:
            import h5py
            
            f = h5py.File(fn, 'w')
            for key in self.history:
                f.create_dataset(key, data=np.array(self.history[key]))
            f.close()
    
        elif suffix == 'npz':
            f = open(fn, 'w')
            np.savez(f, **self.history)
            f.close()
    
        # ASCII format
        else:            
            f = open(fn, 'w')
            print >> f, "#",
    
            for key in self.history:
                print >> f, '%-18s' % key,
    
            print >> f, ''
    
            # Now, the data
            for i in range(len(self.history[key])):
                s = ''
    
                for key in self.sim.history:
                    s += '%-20.8e' % self.history[key][i]
    
                if not s.strip():
                    continue
    
                print >> f, s
    
            f.close()
    
        if os.path.exists('%s.parameters.pkl' % prefix):
            if clobber:
                os.remove('%s.parameters.pkl' % prefix)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        # Save parameter file
        f = open('%s.parameters.pkl' % prefix, 'wb')
        pickle.dump(self.pf, f)
        f.close()
    
        print 'Wrote %s and %s.parameters.pkl' % (fn, prefix)
    