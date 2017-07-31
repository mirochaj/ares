#import os
import numpy as np
#from ..util.ReadData import _sort_history
from ..util import ParameterFile, ProgressBar
#from ..analysis.BlobFactory import BlobFactory
#from ..physics.Constants import nu_0_mhz, E_LyA
from .Global21cm import Global21cm
from ..analysis.PowerSpectrum import PowerSpectrum as analyzePS
#
#try:
#    import dill as pickle
#except ImportError:
#    import pickle

#defaults = \
#{
# 'load_ics': True,
#}

class PowerSpectrum(analyzePS):
    def __init__(self, **kwargs):
        """ Set up a power spectrum calculation. """
        
        self.kwargs = kwargs
        self.snapshots = {}

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf

    @pf.setter
    def pf(self, value):
        self._pf = value

    @property
    def gs(self):
        if not hasattr(self, '_gs'):
            self._gs = Global21cm(**self.kwargs)
        return self._gs
        
    @gs.setter
    def gs(self, value):
        """ Set global 21cm instance by hand. """
        self._gs = value
    
    #@property
    #def global_history(self):
    #    if not hasattr(self, '_global_') 
    
    @property
    def k(self):
        if not hasattr(self, '_k'):
            lkmin = self.pf['powspec_logkmin']
            lkmax = self.pf['powspec_logkmax']
            dlk = self.pf['powspec_dlogk']
            self._logk = np.arange(lkmin, lkmax+dlk, dlk, dtype=float)
            self._k = 10.**self._logk
        return self._k

    def run(self, z):
        """
        Run a simulation over k and (maybe) z.
        
        Returns
        -------
        Nothing: sets `history` attribute.
        
        """
        
        if z in self.snapshots:
            return
                
        self.z = z        
                
        pb = ProgressBar(self.k.size, use=self.pf['progress_bar'])
        
        all_ps = []                        
        for i, (k, data) in enumerate(self.step()):
                          
            if not pb.has_pb:
                pb.start()              
                          
            pb.update(i)
                    
            # Do stuff
            all_ps.append(data)

        pb.finish()
        
        self.snapshots[z] = np.array(all_ps).T
        
    def step(self):
        """
        Generator for the power spectrum.
        """
                                                                                                      
        for k in self.k:
            
            ps_by_pop = []
            
            # Loop over populations
            for i, pop in enumerate(self.gs.pops):
            
                    
                # Check to see if this pop is a source of fluctuations
                # of the desired kind. So far, just Ly-a.
                if not pop.is_lya_src:
                    ps_by_pop.append(0.0)
                    continue
                    
                ps = pop.halos.PowerSpectrum(self.z, k, 
                    profile_FT=lambda z,k,logM: pop.FluxProfileFT(z, k, logM))
        
                ps_by_pop.append(ps)
            
            yield k, ps_by_pop

    #def save(self, prefix, suffix='pkl', clobber=False):
    #    """
    #    Save results of calculation. Pickle parameter file dict.
    #
    #    Notes
    #    -----
    #    1) will save files as prefix.history.suffix and prefix.parameters.pkl.
    #    2) ASCII files will fail if simulation had multiple populations.
    #
    #    Parameters
    #    ----------
    #    prefix : str
    #        Prefix of save filename
    #    suffix : str
    #        Suffix of save filename. Can be hdf5 (or h5), pkl, or npz. 
    #        Anything else will be assumed to be ASCII format (e.g., .txt).
    #    clobber : bool
    #        Overwrite pre-existing files of same name?
    #
    #    """
    #
    #    fn = '%s.snapshot.%s' % (prefix, suffix)
    #
    #    if os.path.exists(fn):
    #        if clobber:
    #            os.remove(fn)
    #        else: 
    #            raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    #
    #    if suffix == 'pkl':                        
    #        f = open(fn, 'wb')
    #        pickle.dump(self.history, f)
    #        f.close()
    #
    #    elif suffix in ['hdf5', 'h5']:
    #        import h5py
    #        
    #        f = h5py.File(fn, 'w')
    #        for key in self.history:
    #            f.create_dataset(key, data=np.array(self.history[key]))
    #        f.close()
    #
    #    elif suffix == 'npz':
    #        f = open(fn, 'w')
    #        np.savez(f, **self.history)
    #        f.close()
    #
    #    # ASCII format
    #    else:            
    #        f = open(fn, 'w')
    #        print >> f, "#",
    #
    #        for key in self.history:
    #            print >> f, '%-18s' % key,
    #
    #        print >> f, ''
    #
    #        # Now, the data
    #        for i in range(len(self.history[key])):
    #            s = ''
    #
    #            for key in self.history:
    #                s += '%-20.8e' % (self.history[key][i])
    #
    #            if not s.strip():
    #                continue
    #
    #            print >> f, s
    #
    #        f.close()
    #
    #    print 'Wrote %s.history_f.%s' % (prefix, suffix)
    #
    #    write_pf = True
    #    if os.path.exists('%s.parameters.pkl' % prefix):
    #        if clobber:
    #            os.remove('%s.parameters.pkl' % prefix)
    #        else: 
    #            write_pf = False
    #            print 'WARNING: %s.parameters.pkl exists! Set clobber=True to overwrite.' % prefix
    #
    #    if write_pf:
    #        # Save parameter file
    #        f = open('%s.parameters.pkl' % prefix, 'wb')
    #        pickle.dump(self.pf, f)
    #        f.close()
    #
    #        print 'Wrote %s.parameters.pkl' % prefix
    #    
    #
    #