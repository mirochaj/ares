"""

WriteData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 14:57:19 2012

Description: 

"""

import os, types
import numpy as np
from ..physics.Cosmology import Cosmology
from ..physics.Constants import s_per_myr
    
try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class CheckPoints(object):
    def __init__(self, pf=None, grid=None, time_units=s_per_myr,
        dtDataDump=5., dzDataDump=None, logdtDataDump=None, logdzDataDump=None,
        stop_time=100., initial_timestep=1.0, source_lifetime=np.inf,
        final_redshift=None, initial_redshift=None):
        self.pf = pf
        self.data = {}
        self.grid = grid

        self.time_units = time_units
        self.stop_time = stop_time * time_units
        self.source_lifetime = source_lifetime * time_units
        self.initial_timestep = initial_timestep * time_units
        self.initial_redshift = initial_redshift
        self.final_redshift = final_redshift
        
        self.fill = 4
        self.t_basename = 'dd'
        self.z_basename = 'rd'
            
        self.time_dumps = False    
        if dtDataDump is not None:
            self.time_dumps = True
            NDD = max(int(float(stop_time) / float(dtDataDump)), 1)
            self.DDtimes = np.linspace(0., self.stop_time, NDD + 1)
        else:
            self.DDtimes = np.array([self.stop_time])
        
        self.redshift_dumps = False
        if dzDataDump is not None: 
            self.redshift_dumps = True 
            # Ordered in increasing time, decreasing redshift 
            self.DDredshifts = np.linspace(initial_redshift, final_redshift, 
                max(int((initial_redshift - final_redshift) / dzDataDump), 1)\
                + 1)
        else:
            self.DDredshifts = np.array([final_redshift])    
            
        # Set time-based data dump schedule
        self.logdtDD = logdtDataDump
        if logdtDataDump is not None:
            self.logti = np.log10(initial_timestep)
            self.logtf = np.log10(stop_time)
            self.logDDt = time_units * np.logspace(self.logti, self.logtf, 
                int((self.logtf - self.logti) / self.logdtDD) + 1)[0:-1]
                
            self.DDtimes = np.sort(np.concatenate((self.DDtimes, self.logDDt)))
            
        # Set redshift-based data dump schedule
        self.logdzDD = logdzDataDump
        if logdzDataDump is not None:
            self.logzi = np.log10(initial_redshift)
            self.logzf = np.log10(final_redshift)
            self.logDDz = np.logspace(self.logzi, self.logzf, 
                int((self.logzf - self.logzi) / self.logdzDD) + 1)[0:-1]
                
            self.DDredshifts = np.sort(np.concatenate((self.DDredshifts, 
                self.logDDz)))    
            self.DDredshifts = list(self.DDredshifts)
            self.DDredshifts.reverse()
            self.DDredshifts = np.array(self.DDredshifts)
                                
        self.DDtimes = np.unique(self.DDtimes)
        self.DDredshifts_asc = np.unique(self.DDredshifts)

        self.allDD = np.linspace(0, len(self.DDtimes)-1., len(self.DDtimes))
        self.allRD = np.linspace(len(self.DDredshifts)-1., 0,
            len(self.DDredshifts))
            
        self.NDD = len(self.allDD)
        self.NRD = len(self.allRD)
                
        if self.grid is not None:
            self.store_ics(grid.data)
            
    @property
    def final_dd(self):
        if not hasattr(self, '_final_dd'):
            if self.redshift_dumps:
                self._final_dd = self.name(t=max(self.RDtimes[-1], 
                    self.DDtimes[-1]))
            else:
                self._final_dd = self.name(t=self.DDtimes[-1])

        return self._final_dd
        
    def store_ics(self, data):
        """
        Write initial conditions. If redshift dumps wanted, store
        initial dataset as both dd and rd.
        """
        nothing = self.update(data, t=0., z=self.initial_redshift)
            
    def update(self, data, t=None, z=None):
        """
        Store data or don't.  If (t + dt) or (z + dz) passes our next checkpoint,
        return new dt (or dz).
        """
        
        to_write, dump_type = self.write_now(t=t, z=z)
        if to_write:
            tmp = data.copy()
            
            if t is not None:
                tmp.update({'time': t})
            if self.grid.expansion:
                if z is not None:
                    tmp.update({'redshift': z})
            
            if dump_type == 'dd':
                self.data[self.name(t=t)] = tmp
            elif dump_type == 'rd':
                self.data[self.name(z=z)] = tmp
            else:
                self.data[self.name(t=t)] = tmp
                self.data[self.name(z=z)] = tmp
                                
            del tmp
                    
    def write_now(self, t=None, z=None):
        """ May be conflict if this time/redshift corresponds to DD and RD. """
        write = False
        kind = None
        if t is not None:
            if t in self.DDtimes:
                write, kind = True, 'dd'
        if z is not None:
            if z in self.DDredshifts and kind == 'dd':
                write, kind = True, 'both'   
            elif z in self.DDredshifts:
                write, kind = True, 'rd'
            
        return write, kind
        
    def next_dt(self, t, dt):
        """
        Compute next timestep based on when our next data dump is, and
        when the source turns off (if ever).
        """
        
        last_dd = int(self.dd(t=t)[0])
        next_dd = last_dd + 1
        
        if t == self.source_lifetime:
            return self.initial_timestep
        
        src_on_now = t < self.source_lifetime
        src_on_next = (t + dt) < self.source_lifetime
                        
        # If dt won't take us all the way to the next DD, don't modify dt
        if self.dd(t=t+dt)[0] <= next_dd:
            if (src_on_now and src_on_next) or (not src_on_now):
                return dt        
            
        if next_dd <= self.NDD:    
            next_dt = self.DDtimes[next_dd] - t
        else:
            next_dt = self.stop_time - t
        
        src_still_on = (t + next_dt) < self.source_lifetime
            
        if src_on_now and src_still_on or (not src_on_now):
            return next_dt
        
        return self.source_lifetime - t 
        
    def dd(self, t=None, z=None):
        """ What data dump are we at currently? Doesn't have to be integer. """
        if t is not None:
            dd = np.interp(t, self.DDtimes, self.allDD, right = self.NDD)
        else:
            dd = None
        if z is not None:
            rd = np.interp(z, self.DDredshifts_asc, self.allRD, right = self.NRD) 
        else:
            rd = None
            
        return dd, rd

    def name(self, t=None, z=None):
        dd, rd = self.dd(t=t, z=z)
        
        if dd is not None:
            return '{0!s}{1!s}'.format(self.t_basename, str(int(dd)).zfill(self.fill))
        else:
            return '{0!s}{1!s}'.format(self.z_basename, str(int(rd)).zfill(self.fill))
        
