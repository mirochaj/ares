"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import os
import numpy as np
from ..util.PrintInfo import print_sim
from ..util.ReadData import _sort_history
from ..util import ParameterFile, ProgressBar
from ..analysis.BlobFactory import BlobFactory
from ..analysis.Global21cm import Global21cm as AnalyzeGlobal21cm
from ..physics.Constants import nu_0_mhz, E_LyA, E_LL, ev_per_hz, erg_per_ev

try:
    import dill as pickle
except ImportError:
    import pickle

defaults = \
{
 'load_ics': True,
}

class Global21cm(BlobFactory,AnalyzeGlobal21cm):
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.
        
        ..note :: This is essentially a MultiPhaseMedium calculation, except
            the Lyman alpha background and 21-cm background are calculated, 
            and alternative (phenomenological) parameterizations such as a 
            tanh for the ionization, thermal, and LW background evolution, 
            may be used.
            
        """
        # See if this is a tanh model calculation
        is_phenom = self.is_phenom = self._check_if_phenom(**kwargs)

        kwargs.update(defaults)
        if 'problem_type' not in kwargs:
            kwargs['problem_type'] = 101

        self.kwargs = kwargs
        
        # Print info to screen
        if self.pf['verbose'] and self.count == 0:
            print_sim(self)
            
    @property 
    def count(self):
        if not hasattr(self, '_count'):
            self._count = 0
        return self._count
            
    @property
    def info(self):
        print_sim(self)

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf

    @pf.setter
    def pf(self, value):
        self._pf = value

    @property
    def medium(self):
        if not hasattr(self, '_medium'):
            from .MultiPhaseMedium import MultiPhaseMedium
            self._medium = MultiPhaseMedium(**self.kwargs)
        return self._medium

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
            self.all_data_igm[i]['dTb'] = float(Tb)
            self.all_data_igm[i]['Ts'] = Ts
            dTb.append(Tb)
            
        return dTb
        
    def _check_if_phenom(self, **kwargs):
        if not kwargs:
            return False
    
        if ('tanh_model' not in kwargs) and ('gaussian_model' not in kwargs):
            return False
            
        self.is_tanh = False
        self.is_gauss = False    

        if 'tanh_model' in kwargs:
            if kwargs['tanh_model']:
                from ..phenom.Tanh21cm import Tanh21cm as PhenomModel
                self.is_tanh = True
                
        elif 'gaussian_model' in kwargs:
            if kwargs['gaussian_model']:
                from ..phenom.Gaussian21cm import Gaussian21cm as PhenomModel            
                self.is_gauss = True
                
        if (not self.is_tanh) and (not self.is_gauss):
            return False
                
        model = PhenomModel(**kwargs)                
        self.pf = model.pf
            
        if self.pf['output_frequencies'] is not None:
            nu = self.pf['output_frequencies']
            z = nu_0_mhz / nu - 1.
        elif self.pf['output_dz'] is not None:
            z = np.arange(self.pf['final_redshift'] + self.pf['output_dz'],
                self.pf['initial_redshift'], self.pf['output_dz'])[-1::-1]
            nu =  nu_0_mhz / (1. + z)   
        else:
            nu_min = self.pf['output_freq_min']
            nu_max = self.pf['output_freq_max']
            nu_res = self.pf['output_freq_res']
        
            nu = np.arange(nu_min, nu_max, nu_res)
            z = nu_0_mhz / nu - 1.
        
        if self.is_gauss:
            self.history = model(nu, **self.pf)    
        else:
            self.history = model(z, **self.pf)

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

        if not hasattr(self, '_suite'):
            self._suite = []    
        
        tf = self.medium.tf
        self.medium._insert_inits()
                
        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        
        # Lists for data in general
        self.all_t, self.all_z, self.all_data_igm, self.all_data_cgm, \
            self.all_RC_igm, self.all_RC_cgm = \
            self.medium.all_t, self.medium.all_z, self.medium.all_data_igm, \
            self.medium.all_data_cgm, self.medium.all_RCs_igm, self.medium.all_RCs_cgm
        
        # Add zeros for Ja
        for element in self.all_data_igm:
            element['Ja'] = 0.0
            element['Jlw'] = 0.0
        
        # List for extrema-finding    
        self.all_dTb = self._init_dTb()
                                        
        for t, z, data_igm, data_cgm, rc_igm, rc_cgm in self.step():
            
            # Delaying the initialization prevents progressbar from being
            # interrupted by, e.g., PrintInfo calls
            if not pb.has_pb:
                pb.start()
                                                    
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
        self.history['dTb'] = self.history['igm_dTb']
        self.history['Ts'] = self.history['igm_Ts']
        self.history['Ja'] = self.history['igm_Ja']
        self.history['Jlw'] = self.history['igm_Jlw']
        
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
                
        self._count += 1
                
        self._suite.append(self.history.copy())
        
        if self.pf['feedback_LW'] and (not self._is_converged()):
            # Compute JLW for next iteration.
            ztmp = self.history['z'][-1::-1]
            Ja = 10.2 * erg_per_ev * self.history['Ja'][-1::-1] / 1e-21
            
            # Should make 50 -> first_light_redshift
            ok = ztmp < 50
            
            f_Jlw = lambda zz: np.interp(zz, ztmp[ok], Ja[ok])
            
            f_Mmin = lambda z: 2.5 * 1e5 * pow(((1.+z)/26.),-1.5) \
                * (1+6.96*pow(4*np.pi* (f_Jlw(z)),0.47))
            
            kw_orig = self.kwargs.copy()
            
            self.kwargs['pop_Mmin{%i}' % self.pf['feedback_LW']] = f_Mmin
                        
            #self.__delattr__('pf')
            #self.__delattr__('_medium')
            #self.__delattr__('history')
            
            delattr(self, '_pf')
            delattr(self, '_medium')
            delattr(self, 'history')       
            #super(Global21cm, self).__delattr__('history')
                        
            self.__init__(**self.kwargs)
            self.run()
                                
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
            Jlw = 0.0
            for i, pop in enumerate(self.medium.field.pops):
                if not pop.is_lya_src:
                    continue
                                                    
                if not np.any(self.medium.field.solve_rte[i]):
                    Ja += self.medium.field.LymanAlphaFlux(z, popid=i)                    
                    Jlw += self.medium.field.LymanWernerFlux(z, popid=i)
                    continue

                # Grab line fluxes for this population for this step
                for j, band in enumerate(self.medium.field.bands_by_pop[i]):
                    E0, E1 = band
                    if not (E0 <= E_LyA < E1):
                        continue
                    
                    Earr = np.concatenate(self.medium.field.energies[i][j])
                    l = np.argmin(np.abs(Earr - E_LyA))     # should be 0
                    
                    Ja += self.medium.field.all_fluxes[-1][i][j][l]

                    ##
                    # Feedback time
                    ##
                    
                    # Find photons in LW band    
                    is_LW = np.logical_and(Earr >= 11.18, Earr <= E_LL)
                    
                    # And corresponding fluxes
                    flux = self.medium.field.all_fluxes[-1][i][j][is_LW]
                    
                    # Convert to energy units, and per eV to prep for integral
                    flux *= Earr[is_LW] * erg_per_ev / ev_per_hz
                    
                    dnu = (E_LL - 11.18) / ev_per_hz
                    Jlw += np.trapz(flux, x=Earr[is_LW]) / dnu
                                        
            # Solver requires this                                            
            Ja = np.atleast_1d(Ja)                                            
                                                                    
            # Compute spin temperature
            n_H = self.medium.parcel_igm.grid.cosm.nH(z)
            Ts = self.medium.parcel_igm.grid.hydr.Ts(z,
                data_igm['Tk'], Ja, data_igm['h_2'], data_igm['e'] * n_H)

            # Compute volume-averaged ionized fraction
            xavg = data_cgm['h_2'] + (1. - data_cgm['h_2']) * data_igm['h_2']

            # Derive brightness temperature
            dTb = self.medium.parcel_igm.grid.hydr.dTb(z, xavg, Ts)

            # Add derived fields to data
            data_igm.update({'Ts': Ts, 'dTb': dTb, 'Ja': Ja, 'Jlw': Jlw})
                        
            # Apply LW feedback
            for i, pop in enumerate(self.medium.field.pops):
                                
                if not pop.pf['pop_feedback']:
                    continue
                    
                # Compute new minimum mass
                new_Mmin = 2.5 * 1e5 * pow(((1.+z)/26.),-1.5) \
                    * (1+6.96*pow(4*np.pi* (Jlw / 1e-21),0.47)) 
                                                                                                        
                # Reset the minimum mass of star-forming halos
                pop.update_Mmin(z, new_Mmin)
                        
            # Yield!            
            yield t, z, data_igm, data_cgm, RC_igm, RC_cgm 
            
    def _is_converged(self):
        # Perform set number of iterations
        if self.pf['feedback_LW_iter'] is not None:
            if self.count > self.pf['feedback_LW_iter']:
                return True
        elif self.count > self.pf['feedback_LW_maxiter']:
            return True
        # Iterate until convergence criterion is met
        elif self.count > 1:
    
            # Grab global spectra for this iteration and previous one
            znow = self.history['z'][-1::-1]
            dTb_now = self.history['dTb'][-1::-1]
            
            sim_pre = self._suite[-2]
            zpre = sim_pre['z'][-1::-1]
            dTb_pre = sim_pre['dTb'][-1::-1]
            
            ok = znow < 50
            
            # Interpolate to common redshift grid
            dTb_now_interp = np.interp(zpre, znow, dTb_now)
            
            dTb_pre = dTb_pre[np.where(ok)]
            dTb_now_interp = dTb_now_interp[np.where(ok)]
            
            err_rel = np.abs((dTb_now_interp - dTb_pre) \
                / dTb_now_interp)
            err_abs = np.abs(dTb_now_interp - dTb_pre)
            
            #print "Iteration #%i: err_rel (mean) = %.5g, err_abs (mean) = %.5g" \
            #    % (self.count, err_rel.mean(), err_abs.mean())
            #print "Iteration #%i: err_rel (max) = %.5g, err_abs (max) = %.5g" \
            #    % (self.count, err_rel.max(), err_abs.max()) 
            
            rtol = self.pf['feedback_LW_rtol']
            atol = self.pf['feedback_LW_atol']

            # Compute error
            if self.pf['feedback_LW_mean_err']:
                
                
                if rtol > 0:
                    if err_rel.mean() > rtol:
                        return False
                    elif err_rel.mean() < rtol and (atol == 0):
                        return True
                    
                # Only make it here if rtol is satisfied or irrelevant
                
                if atol > 0:
                    if err_abs.mean() < atol:
                        return True                        
                
            else:  
                converged = np.allclose(dTb_pre, dTb_now_interp,
                    rtol=rtol, atol=atol)
                                          
                if converged:
                    return True
        else:
            pass
            
            # This should only happen on iteration #1 when we're using
            # a tolerance-based convergence criterion.

        return False

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
    
        fn = '%s.history.%s' % (prefix, suffix)
    
        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        if suffix == 'pkl':                        
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
    
                for key in self.history:
                    s += '%-20.8e' % (self.history[key][i])
    
                if not s.strip():
                    continue
    
                print >> f, s
    
            f.close()
    
        print 'Wrote %s.history.%s' % (prefix, suffix)
    
        write_pf = True
        if os.path.exists('%s.parameters.pkl' % prefix):
            if clobber:
                os.remove('%s.parameters.pkl' % prefix)
            else: 
                write_pf = False
                print 'WARNING: %s.parameters.pkl exists! Set clobber=True to overwrite.' % prefix

        if write_pf:
            # Save parameter file
            f = open('%s.parameters.pkl' % prefix, 'wb')
            pickle.dump(self.pf, f)
            f.close()
    
            print 'Wrote %s.parameters.pkl' % prefix
        
    
