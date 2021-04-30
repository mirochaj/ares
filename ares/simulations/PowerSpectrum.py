#import os
import numpy as np
#from ..util.ReadData import _sort_history
from ..util import ParameterFile, ProgressBar
#from ..analysis.BlobFactory import BlobFactory
#from ..physics.Constants import nu_0_mhz, E_LyA
from .Global21cm import Global21cm
from ..analysis.PowerSpectrum import PowerSpectrum as analyzePS
from ..physics.Constants import cm_per_mpc, c, s_per_yr, erg_per_ev, \
    erg_per_s_per_nW, h_p, cm_per_m

#
#try:
#    import dill as pickle
#except ImportError:
#    import pickle

#defaults = \
#{
# 'load_ics': True,
#}

class PowerSpectrum(analyzePS): # pragma: no cover
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
                if not pop.is_src_lya:
                    ps_by_pop.append(0.0)
                    continue

                ps = pop.halos.PowerSpectrum(self.z, k,
                    profile_FT=lambda z,k,logM: pop.FluxProfileFT(z, k, logM))

                ps_by_pop.append(ps)

            yield k, ps_by_pop
