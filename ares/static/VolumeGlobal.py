"""

IntergalacticMedium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May 24 11:31:06 2013

Description: 

"""

import numpy as np
from ..util.Warnings import *
from ..util import ProgressBar
from ..physics.Constants import *
import types, os, re, sys, pickle
from ..util.Misc import num_freq_bins
from ..physics import SecondaryElectrons
from scipy.integrate import dblquad, romb, simps, quad, trapz
from ..util.Warnings import tau_tab_z_mismatch, tau_tab_E_mismatch

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

log10 = np.log(10.)
E_th = np.array([13.6, 24.4, 54.4])

defkwargs = \
{
 'zf':None, 
 'xray_flux':None, 
 'epsilon_X': None,
 'Gamma': None,
 'gamma': None,
 'return_rc': False, 
 'energy_units':False, 
 'Emax': None,
 #'zxavg':0.0,
 #'igm':True,
 'xavg': 0.0,
 'igm_h_1': 1.0,
 'igm_h_2': 0.0,
 'igm_he_2': 0.0,
 'igm_he_3': 0.0,
 'cgm_h_1': 1.0,
 'cgm_h_2': 0.0,
 'cgm_he_2': 0.0,
 'cgm_he_3': 0.0,
 'igm_e': 0.0,
}

species_i_to_str = {0:'h_1', 1:'he_1', 2:'he_2'}

class GlobalVolume(object):
    def __init__(self, background):
        """
        Initialize a GlobalVolume.
        
        Parameters
        ----------
        background : ares.solvers.UniformBackground instance.
        
        """

        self.background = background
        self.pf = background.pf
        self.grid = background.grid
        self.cosm = background.cosm
        self.hydr = background.hydr
        self.pops = background.pops
        self.Npops = len(self.pops)
        
        # Include helium opacities approximately?
        self.approx_He = self.pf['include_He'] and self.pf['approx_He']
        
        # Include helium opacities self-consistently?
        self.self_consistent_He = self.pf['include_He'] \
            and (not self.pf['approx_He'])

        self.esec = \
            SecondaryElectrons(method=self.pf["secondary_ionization"])  

        # Choose function for computing bound-free absorption cross-sections                
        if self.pf['approx_sigma']:
            from ..physics.CrossSections import \
                ApproximatePhotoIonizationCrossSection as sigma
        else:
            from ..physics.CrossSections import \
                PhotoIonizationCrossSection as sigma

        self.sigma = sigma
        self.sigma0 = sigma(E_th[0])    # Hydrogen ionization threshold

        self._set_integrator()

    @property
    def rates_no_RT(self):
        if not hasattr(self, '_rates_no_RT'):
            self._rates_no_RT = \
                {'k_ion': np.zeros((self.grid.dims,
                    self.grid.N_absorbers)),
                 'k_heat': np.zeros((self.grid.dims,
                    self.grid.N_absorbers)),
                 'k_ion2': np.zeros((self.grid.dims,
                    self.grid.N_absorbers, self.grid.N_absorbers)),
                }

        return self._rates_no_RT

    #def _fetch_tau(self, pop, zpf, Epf):
    #    """
    #    Look for optical depth tables. Supply corrected energy and redshift
    #    arrays if there is a mistmatch between those generated from information
    #    in the parameter file and those found in the optical depth table.
    #    
    #    .. note:: This will only be called from UniformBackground, and on
    #        populations which are using the generator framework.
    #    
    #    Parameters
    #    ----------
    #    popid : int
    #        ID # for population of interest.
    #    zpf : np.ndarray
    #        What the redshifts should be according to the parameter file.    
    #    Epf : np.ndarray
    #        What the energies should be according to the parameter file.
    #    
    #    Returns
    #    -------
    #    Energies and redshifts, potentially revised from Epf and zpf.
    #    
    #    """
    #    
    #    for i in range(self.Npops):
    #        if pop == self.pops[i]:
    #            band = self.background.bands_by_pop[i]
    #            break
    #        
    #    # First, look in CWD or $ARES (if it exists)
    #    self.tabname = self._load_tau(pop, pop.pf['tau_prefix'])
    #            
    #    if not self.tabname:
    #        return zpf, Epf, None
    #    
    #    # If we made it this far, we found a table that may be suitable
    #    ztab, Etab, tau = self._read_tau(self.tabname)
    #            
    #    # Return right away if there's no potential for conflict
    #    if (zpf is None) and (Epf is None):
    #        return ztab, Etab, tau
    #        
    #    # Figure out if the tables need fixing    
    #    zmax_ok = \
    #        (ztab.max() >= zpf.max()) or \
    #        np.allclose(ztab.max(), zpf.max())
    #    zmin_ok = \
    #        (ztab.min() <= zpf.min()) or \
    #        np.allclose(ztab.min(), zpf.min())
    #                            
    #    Emin_ok = \
    #        (Etab.min() <= Epf.min()) or \
    #        np.allclose(Etab.min(), Epf.min())
    #    
    #    # Results insensitive to Emax (so long as its relatively large)
    #    # so be lenient with this condition (100 eV or 1% difference
    #    # between parameter file and lookup table)
    #    Emax_ok = np.allclose(Etab.max(), Epf.max(), atol=100., rtol=1e-2)
    #            
    #    # Check redshift bounds
    #    if not (zmax_ok and zmin_ok):
    #        if not zmax_ok:
    #            tau_tab_z_mismatch(self, zmin_ok, zmax_ok, ztab)
    #            sys.exit(1)
    #        else:
    #            if self.pf['verbose']:
    #                tau_tab_z_mismatch(self, zmin_ok, zmax_ok, ztab)
    #    
    #    if not (Emax_ok and Emin_ok):
    #        if self.pf['verbose']:
    #            tau_tab_E_mismatch(pop, self.tabname, Emin_ok, Emax_ok, Etab)
    #            
    #        if Etab.max() < Epf.max():
    #            sys.exit(1)
    #                                                        
    #    # Correct for inconsistencies between parameter file and table
    #    # By effectively masking out those elements with tau -> inf
    #    if Epf.min() > Etab.min():
    #        Ediff = Etab - Epf.min()
    #        i_E0 = np.argmin(np.abs(Ediff))
    #        if Ediff[i_E0] < 0:
    #            i_E0 += 1
    #    
    #        #tau[:,0:i_E0+1] = np.inf
    #    else:
    #        i_E0 = 0
    #    
    #    if Epf.max() < Etab.max():
    #        Ediff = Etab - Epf.max()
    #        i_E1 = np.argmin(np.abs(Ediff))
    #        if Ediff[i_E1] < 0:
    #            i_E1 += 1
    #    
    #        #tau[:,i_E1+1:] = np.inf
    #    else:
    #        i_E1 = None
    #        
    #    # We're done!
    #    return ztab, Etab[i_E0:i_E1], tau[:,i_E0:i_E1]

    @property
    def E(self):
        if not hasattr(self, '_E'):
            self._tabulate_atomic_data()
    
        return self._E

    @property
    def sigma_E(self):
        if not hasattr(self, '_sigma_E'):
            self._tabulate_atomic_data()
        
        return self._sigma_E
        
    def _tabulate_atomic_data(self):
        """
        Pre-compute cross sections and such for each source population.
        
        Returns
        -------
        Nothing. Sets the following attributes:
        
        sigma_E
        log_sigma_E
        fheat, flya, fion
        
        """

        # Remember: this is [Npops, Nbands/pop]
        self._E = self.background.energies
        
        self.logE = [[] for k in range(self.Npops)]
        self.dlogE = [[] for k in range(self.Npops)]
        self._sigma_E = [[] for k in range(self.Npops)]
        self.fheat = [[] for k in range(self.Npops)]
        self.flya = [[] for k in range(self.Npops)]
        
        # Semi-special treatment for secondary ionization
        #self.fion = [{} for k in range(self.Npops)]
        
        # Loop over populations
        for i, pop in enumerate(self.pops):
                        
            # This means the population is completely approximate
            if not self.background.solve_rte[i]:
                self.logE[i] = [None]
                self.dlogE[i] = [None]
                self._sigma_E[i] = [None]
                self.fheat[i] = [None]
                #self.fion[i] = [None]
                self.flya[i] = [None]
                continue
            else:
                Nbands = len(self.background.energies[i])

                self.logE[i] = [None for k in range(Nbands)]
                self.dlogE[i] = [None for k in range(Nbands)]
                self._sigma_E[i] = [None for k in range(Nbands)]
                self.fheat[i] = [None for k in range(Nbands)]
                #self.fion[i] = [None for k in range(Nbands)]
                self.flya[i] = [None for k in range(Nbands)]
                
            # If we make it here, the population has at least one band that
            # requires a detailed solution to the RTE    
                
            # Loop over each band for this population
            for j, band in enumerate(self.background.energies[i]):
            
                if band is None:
                    continue

                # More convenient variables
                E = self._E[i][j]    
                N = E.size

                # Compute some things we need, like bound-free cross-section
                self.logE[i][j] = np.log10(E)
                self.dlogE[i][j] = np.diff(self.logE[i][j])
                self._sigma_E[i][j] = \
                    np.array([np.array(map(lambda E: self.sigma(E, k), E)) \
                    for k in xrange(3)])

                # Pre-compute secondary ionization and heating factors
                if self.esec.Method > 1:
                
                    self.fheat[i][j] = np.ones([N, len(self.esec.x)])
                    self.flya[i][j] = np.ones([N, len(self.esec.x)])
                
                    #self.fion[popid] = {}
                    #self.fion[popid]['h_1'] = np.ones([N, len(self.esec.x)])
                
                    # Must evaluate at ELECTRON energy, not photon energy
                    for i, nrg in enumerate(E - E_th[0]):
                        self.fheat[popid][i,:] = \
                            self.esec.DepositionFraction(self.esec.x, E=nrg, 
                            channel='heat')
                        self.fion[popid]['h_1'][i,:] = \
                            self.esec.DepositionFraction(self.esec.x, E=nrg, 
                            channel='h_1')
                
                        if self.pf['secondary_lya']:
                            self.flya[popid][i,:] = \
                                self.esec.DepositionFraction(self.esec.x, E=nrg, 
                                channel='lya') 
                
                    # Helium
                    if self.pf['include_He'] and not self.pf['approx_He']:
                
                        self.fion[popid]['he_1'] = np.ones([N, len(self.esec.x)])
                        self.fion[popid]['he_2'] = np.ones([N, len(self.esec.x)])
                
                        for i, nrg in enumerate(E - E_th[1]):
                            self.fion[popid]['he_1'][i,:] = \
                                self.esec.DepositionFraction(self.esec.x, 
                                E=nrg, channel='he_1')
                
                        for i, nrg in enumerate(E - E_th[2]):
                            self.fion[popid]['he_2'][i,:] = \
                                self.esec.DepositionFraction(self.esec.x, 
                                E=nrg, channel='he_2')    
                
        return        
                
        #self.logE = [np.log10(nrg) for nrg in self._E]
        #self.dlogE = [np.diff(nrgs) for nrgs in self.logE]
        #
        #self._sigma_E = [None for i in range(self.Npops)]
        #self.fheat = [None for i in range(self.Npops)]
        #self.fion = [None for i in range(self.Npops)]
        #self.flya = [None for i in range(self.Npops)]

        # Loop over source populations
        for popid, src in enumerate(self.pops):

            if not self.background.solve_rte[popid]:
                continue

            E = self._E[popid]
            N = E.size

            # Pre-compute cross-sections
            self._sigma_E[popid] = \
                np.array([np.array(map(lambda E: self.sigma(E, i), E)) \
                for i in xrange(3)])

            # Pre-compute secondary ionization and heating factors
            if self.esec.Method > 1:

                self.fheat[i] = np.ones([N, len(self.esec.x)])
                self.flya[i] = np.ones([N, len(self.esec.x)])

                self.fion[i] = {}
                self.fion[i]['h_1'] = np.ones([N, len(self.esec.x)])

                # Must evaluate at ELECTRON energy, not photon energy
                for k, nrg in enumerate(E - E_th[0]):
                    self.fheat[i][k,:] = \
                        self.esec.DepositionFraction(self.esec.x, E=nrg, 
                        channel='heat')
                    self.fion[i]['h_1'][k,:] = \
                        self.esec.DepositionFraction(self.esec.x, E=nrg, 
                        channel='h_1')

                    if self.pf['secondary_lya']:
                        self.flya[i][k,:] = \
                            self.esec.DepositionFraction(self.esec.x, E=nrg, 
                            channel='lya') 
                        
                # Helium
                if self.pf['include_He'] and not self.pf['approx_He']:
                    
                    self.fion[i]['he_1'] = np.ones([N, len(self.esec.x)])
                    self.fion[i]['he_2'] = np.ones([N, len(self.esec.x)])
                    
                    for k, nrg in enumerate(E - E_th[1]):
                        self.fion[popid]['he_1'][k,:] = \
                            self.esec.DepositionFraction(self.esec.x, 
                            E=nrg, channel='he_1')
                    
                    for k, nrg in enumerate(E - E_th[2]):
                        self.fion[popid]['he_2'][k,:] = \
                            self.esec.DepositionFraction(self.esec.x, 
                            E=nrg, channel='he_2')            
                            
    def _set_integrator(self):
        self.integrator = self.pf["unsampled_integrator"]
        self.sampled_integrator = self.pf["sampled_integrator"]
        self.rtol = self.pf["integrator_rtol"]
        self.atol = self.pf["integrator_atol"]
        self.divmax = int(self.pf["integrator_divmax"])
    
    #def _read_tau(self, fn):
    #    """ Read optical depth table. """
    #    
    #    if type(fn) is dict:
    #        
    #        E0 = fn['E'].min()
    #        E1 = fn['E'].max()
    #        E = fn['E']
    #        z = fn['z']
    #        x = z + 1
    #        N = E.size
    #            
    #        R = x[1] / self.x[0]
    #        
    #        tau = fn['tau']
    #
    #    elif re.search('hdf5', fn):
    #
    #        f = h5py.File(self.tabname, 'r')
    #
    #        E0 = min(f['photon_energy'].value)
    #        E1 = max(f['photon_energy'].value)
    #        E = f['photon_energy'].value
    #        z = f['redshift'].value
    #        x = z + 1
    #        N = E.size
    #            
    #        R = x[1] / x[0]
    #        
    #        tau = f['tau'].value
    #        f.close()
    #
    #    elif re.search('npz', fn) or re.search('pkl', fn):    
    #
    #        if re.search('pkl', fn):
    #            f = open(fn, 'rb')
    #            data = pickle.load(f)
    #        else:
    #            f = open(fn, 'r')
    #            data = dict(np.load(f))
    #        
    #        E0 = data['E'].min()
    #        E1 = data['E'].max()            
    #        E = data['E']
    #        z = data['z']
    #        x = z + 1
    #        N = E.size
    #        
    #        R = x[1] / x[0]
    #        
    #        tau = tau = data['tau']
    #        f.close()
    #    else:
    #        raise NotImplemented('Don\'t know how to read %s.' % fn)
    #
    #    return z, E, tau
    
    #def _tau_name(self, pop, suffix='hdf5'):
    #    """
    #    Return name of table based on its properties.
    #    """
    #
    #    if not have_h5py:
    #        suffix == 'pkl'
    #
    #    HorHe = 'He' if self.pf['include_He'] else 'H'
    #
    #    zf = self.pf['final_redshift']
    #    zi = self.pf['initial_redshift']
    #
    #    L, N = self._tau_shape(pop)
    #
    #    E0 = pop.pf['pop_Emin']
    #    E1 = pop.pf['pop_Emax']
    #
    #    fn = lambda z1, z2, E1, E2: \
    #        'optical_depth_%s_%ix%i_z_%i-%i_logE_%.2g-%.2g.%s' \
    #        % (HorHe, L, N, z1, z2, E1, E2, suffix)
    #
    #    return fn(zf, zi, np.log10(E0), np.log10(E1)), fn
    
    #def _load_tau(self, pop, prefix=None):
    #    """
    #    Find an optical depth table.
    #    """
    #    
    #    fn, fn_func = self._tau_name(pop)
    #
    #    if prefix is None:
    #        ares_dir = os.environ.get('ARES')
    #        if not ares_dir:
    #            print "No ARES environment variable."
    #            return None
    #        
    #        input_dirs = [os.path.join(ares_dir,'input','optical_depth')]
    #
    #    else:
    #        if type(prefix) is str:
    #            input_dirs = [prefix]
    #        else:
    #            input_dirs = prefix
    #
    #    guess = os.path.join(input_dirs[0], fn)
    #    if os.path.exists(guess):
    #        return guess
    #
    #    ## Find exactly what table should be
    #    zmin, zmax, Nz, lEmin, lEmax, chem, pre, post = self._parse_tab(fn)
    #
    #    ok_matches = []
    #    perfect_matches = []
    #    
    #    # Loop through input directories
    #    for input_dir in input_dirs:
    #                        
    #        # Loop over files in input_dir, look for best match
    #        for fn1 in os.listdir(input_dir):
    #            
    #            if re.search('hdf5', fn1) and (not have_h5py):
    #                continue
    #
    #            tab_name = os.path.join(input_dir, fn1)
    #            
    #            try:
    #                zmin_f, zmax_f, Nz_f, lEmin_f, lEmax_f, chem_f, p1, p2 = \
    #                    self._parse_tab(fn1)
    #            except:
    #                continue
    #
    #            # Dealbreakers
    #            if Nz_f != Nz:
    #                continue
    #            if zmax_f < zmax:
    #                continue
    #            if chem_f != chem:
    #                continue
    #
    #            # Continue with possible matches
    #            for fmt in ['pkl', 'npz', 'hdf5']:
    #
    #                if fn1 == fn and fmt == self.pf['preferred_format']:
    #                    perfect_matches.append(tab_name)
    #                    continue
    #
    #                if c and fmt == self.pf['preferred_format']:
    #                    perfect_matches.append(tab_name)
    #                    continue
    #
    #                # If number of redshift bins and energy range right...
    #                if re.search(pre, fn1) and re.search(post, fn1):
    #                    if re.search(fmt, fn1) and fmt == self.pf['preferred_format']:
    #                        perfect_matches.append(tab_name)
    #                    else:
    #                        ok_matches.append(tab_name)
    #                
    #                # If number of redshift bins is right...
    #                elif re.search(pre, fn1):
    #                                            
    #                    if re.search(fmt, fn1) and fmt == self.pf['preferred_format']:
    #                        perfect_matches.append(tab_name)
    #                    else:
    #                        ok_matches.append(tab_name)
    #    
    #    if perfect_matches:
    #        return perfect_matches[0]
    #    elif ok_matches:
    #        return ok_matches[0]
    #    else:
    #        return None
            
    #def _parse_tab(self, fn):
    #            
    #    tmp1, tmp2 = fn.split('_z_')
    #    pre = tmp1[0:tmp1.rfind('x')]
    #    red, tmp3 = fn.split('_logE_')
    #    post = '_logE_' + tmp3.replace('.hdf5', '')
    #    
    #    # Find exactly what table should be
    #    zmin, zmax = map(float, red[red.rfind('z')+2:].partition('-')[0::2])
    #    logEmin, logEmax = map(float, tmp3[tmp3.rfind('E')+1:tmp3.rfind('.')].partition('-')[0::2])
    #    
    #    Nz = pre[pre.rfind('_')+1:]
    #    
    #    # Hack off Nz string and optical_depth_
    #    chem = pre.strip(Nz)[14:-1]#.strip('optical_depth_')
    #    
    #    return zmin, zmax, int(Nz), logEmin, logEmax, chem, pre, post
    #            
    #def _tau_shape(self, pop):
    #    """
    #    Determine dimensions of optical depth table.
    #    
    #    Unfortunately, this is a bit redundant with the procedure in
    #    self._init_xrb, but that's the way it goes.
    #    """
    #    
    #    # Set up log-grid in parameter x = 1 + z
    #    x = np.logspace(np.log10(1+self.pf['final_redshift']),
    #        np.log10(1+self.pf['initial_redshift']),
    #        int(pop.pf['pop_tau_Nz']))
    #    z = x - 1.
    #    logx = np.log10(x)
    #    logz = np.log10(z)
    #
    #    # Constant ratio between elements in x-grid
    #    R = x[1] / x[0]
    #    logR = np.log10(R)
    #    
    #    E0 = pop.pf['pop_Emin']
    #    
    #    # Create mapping to frequency space
    #    E = 1. * E0
    #    n = 1
    #    while E < pop.pf['pop_Emax']:
    #        E = E0 * R**(n - 1)
    #        n += 1    
    #    
    #    # Set attributes for dimensions of optical depth grid
    #    L = len(x)
    #    
    #    # Frequency grid must be index 1-based.
    #    N = num_freq_bins(L, zi=self.pf['initial_redshift'], 
    #        zf=self.pf['final_redshift'], Emin=E0, 
    #        Emax=pop.pf['pop_Emax'])
    #    N -= 1
    #    
    #    return L, N
    
    def RestFrameEnergy(self, z, E, zp):
        """
        Return energy of a photon observed at (z, E) and emitted at zp.
        """
        
        return E * (1. + zp) / (1. + z)
    
    def ObserverFrameEnergy(self, z, Ep, zp):
        """
        What is the energy of a photon observed at redshift z and emitted 
        at redshift zp and energy Ep?
        """

        return Ep * (1. + z) / (1. + zp)

    def Jc(self, z, E):
        """
        Flux corresponding to one photon per hydrogen atom at redshift z.
        """

        return c * self.cosm.nH0 * (1. + z)**3 / 4. / np.pi \
            / (E * erg_per_ev / h)

    def rate_to_coefficient(self, z, species=0, zone='igm', **kw):
        """
        Convert an ionization/heating rate to a rate coefficient.
        
        Provides units of per atom.
        """
        
        if self.pf['photon_counting']:
            prefix = zone
        else:
            prefix = 'igm'
        
        if species == 0:     
            weight = 1. / self.cosm.nH(z) / kw['%s_h_1' % prefix]
        elif species == 1:
            weight = 1. / self.cosm.nHe(z) / kw['%s_he_1' % prefix]
        elif species == 2:
            weight = 1. / self.cosm.nHe(z) / kw['%s_he_2' % prefix]

        return weight

    def coefficient_to_rate(self, z, species=0, **kw):
        return 1. / self.rate_to_coefficient(z, species, **kw)

    def _fix_kwargs(self, functionify=False, popid=0, band=0, **kwargs):

        kw = defkwargs.copy()
        kw.update(kwargs)

        pop = self.pops[popid]

        if functionify and type(kw['xavg']) is not types.FunctionType:
            tmp = kw['xavg']
            kw['xavg'] = lambda z: tmp

        if kw['zf'] is None and pop is not None:
            kw['zf'] = pop.zform
        
        if not self.background.solve_rte[popid]:
            pass
        elif (kw['Emax'] is None) and self.background.solve_rte[popid] and \
            np.any(self.background.bands_by_pop[popid] > pop.pf['pop_EminX']):
            
            kw['Emax'] = self.background.energies[popid][band][-1]
                        
        return kw
        
    def HeatingRate(self, z, species=0, popid=0, band=0, **kwargs):
        """
        Compute heating rate density due to emission from this population. 
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        species : int
            Atom whose liberated electrons cause heating.
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)
        
        ===============
        relevant kwargs
        ===============
        xray_flux : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of 
            erg s**-1 cm**-3 atom**-1.    
        
        Returns
        -------
        Proper heating rate density in units of in erg s**-1 cm**-3 at redshift z,
        due to electrons previously bound to input species.

        """
                
        pop = self.pops[popid]
                                
        if not pop.pf['pop_heat_src_igm'] or (z >= pop.zform):
            return 0.0        
                
        # Grab defaults, do some patches if need be    
        kw = self._fix_kwargs(**kwargs)

        if pop.pf['pop_k_heat_igm'] is not None:
            return pop.pf['pop_k_heat_igm'](z)

        # Compute fraction of photo-electron energy deposited as heat
        if pop.pf['pop_fXh'] is None:
            
            # Interpolate in energy and ionized fraction
            if self.esec.Method > 1 and (kw['fluxes'][popid][band] is not None):
                if kw['igm_e'] <= self.esec.x[0]:
                    fheat = self.fheat[popid][:,0]
                else:

                    i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                    if self.esec.x[i_x] > kw['igm_e']:
                        i_x -= 1
                        
                    j = i_x + 1    
                    
                    fheat = self.fheat[popid][:,i_x] \
                        + (self.fheat[popid][:,j] - self.fheat[popid][:,i_x]) \
                        * (kw['igm_e'] - self.esec.x[i_x]) \
                        / (self.esec.x[j] - self.esec.x[i_x])                
            else:
                fheat = self.esec.DepositionFraction(kw['igm_e'])[0]

        else:
            fheat = self.pf['pop_fXh']

        # Assume heating rate density at redshift z is only due to emission
        # from sources at redshift z
        if not self.background.solve_rte[popid]:
            weight = self.rate_to_coefficient(z, species, **kw)
            
            Lx = pop.LuminosityDensity(z, Emin=pop.pf['pop_Emin_xray'], 
                Emax=pop.pf['pop_Emax'])
                                                
            return weight * fheat * Lx * (1. + z)**3
            
        # Otherwise, do the full calculation

        # Re-normalize to help integrator
        norm = J21_num * self.sigma0
                
        # Computes excess photo-electron energy due to ionizations by
        # photons with energy E (normalized by sigma0 * Jhat)
        if kw['fluxes'][popid] is None:

            # If we're approximating helium, must add contributions now
            # since we'll never explicitly call this method w/ species=1.
            if self.approx_He:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg']) \
                    * (self.sigma(E) * (E - E_th[0]) \
                    + self.cosm.y * self.sigma(E, species=1) * (E - E_th[1])) \
                    * fheat / norm / ev_per_hz
                    
            # Otherwise, just heating via hydrogen photo-electrons
            else:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'], 
                    zxavg=kw['zxavg']) * self.sigma(E, species=1) \
                    * (E - E_th[species]) * fheat / norm / ev_per_hz
        
        # This means the fluxes have been computed already - integrate
        # over discrete set of points
        else:
            
            integrand = self.sigma_E[popid][band][species] \
                * (self._E[popid][band] - E_th[species])

            if self.approx_He:
                integrand += self.cosm.y * self.sigma_E[popid][band][1] \
                    * (self._E[popid][band] - E_th[1])
                    
            integrand *= kw['fluxes'][popid][band] * fheat / norm / ev_per_hz
                         
        # Compute integral over energy
        if type(integrand) == types.FunctionType:
            heat, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0, 
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)
        else:
            if kw['Emax'] is not None:
                imax = np.argmin(np.abs(self._E[popid][band] - kw['Emax']))
                if imax == 0:
                    return 0.0
                elif imax == (len(self._E[popid][band]) - 1):  
                    imax = None 
                                        
                if self.sampled_integrator == 'romb':
                    raise ValueError("Romberg's method cannot be used for integrating subintervals.")
                    heat = romb(integrand[0:imax] * self.E[0:imax], 
                        dx=self.dlogE[0:imax])[0] * log10
                else:
                    heat = simps(integrand[0:imax] * self._E[popid][band][0:imax], 
                        x=self.logE[popid][band][0:imax]) * log10
            
            else:
                imin = np.argmin(np.abs(self._E[popid][band] - pop.pf['pop_Emin']))
                
                if self.sampled_integrator == 'romb':
                    heat = romb(integrand[imin:] * self._E[popid][band][imin:], 
                        dx=self.dlogE[popid][band][imin:])[0] * log10
                elif self.sampled_integrator == 'trapz':
                    heat = np.trapz(integrand[imin:] * self._E[popid][band][imin:], 
                        x=self.logE[popid][band][imin:]) * log10
                else:
                    heat = simps(integrand[imin:] * self._E[popid][band][imin:], 
                        x=self.logE[popid][band][imin:]) * log10
          
        # Re-normalize, get rid of per steradian units
        heat *= 4. * np.pi * norm * erg_per_ev

        # Currently a rate coefficient, returned value depends on return_rc                                      
        if kw['return_rc']:
            pass
        else:
            heat *= self.coefficient_to_rate(z, species, **kw)

        return heat    
        
    def IonizationRateCGM(self, z, species=0, popid=0, **kwargs):
        """
        Compute growth rate of HII regions.

        Parameters
        ----------
        z : float
            current redshift
        species : int
            Ionization rate for what atom?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)
            
        ===============
        relevant kwargs
        ===============
        fluxes : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of 
            erg s**-1 cm**-3 atom**-1.    

        Returns
        -------
        Ionization rate. Units determined by value of return_rc keyword
        argument, which is False by default.

        """
        
        pop = self.pops[popid]
        
        if (not pop.pf['pop_ion_src_cgm']) or (z > pop.zform):
            return 0.0
            
        # Need some guidance from 1-D calculations to do this
        if species > 0:
            return 0.0
        
        kw = defkwargs.copy()
        kw.update(kwargs)

        if pop.pf['pop_k_ion_cgm'] is not None:
            return self.pf['pop_k_ion_cgm'](z)
                
        if kw['return_rc']:
            weight = self.rate_to_coefficient(z, species, **kw)
        else:
            weight = 1.0
          
        Qdot = pop.PhotonLuminosityDensity(z, Emin=13.6, Emax=24.6)
                        
        return weight * Qdot * (1. + z)**3  
            
    def IonizationRateIGM(self, z, species=0, popid=0, band=0, **kwargs):
        """
        Compute volume averaged hydrogen ionization rate.
        
        Parameters
        ----------
        z : float
            redshift
        species : int
            HI, HeI, or HeII (species=0, 1, 2, respectively)
            
        Returns
        -------
        Volume averaged ionization rate in units of ionizations per 
        second. If return_rc=True, will be in units of ionizations per
        second per atom.
        
        """

        pop = self.pops[popid]

        # z between zform, zdead? must be careful for BHs
        if (not pop.pf['pop_ion_src_igm']) or (z > pop.zform):
            return 0.0

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(**kwargs)

        if pop.pf['pop_k_ion_igm'] is not None:
            return pop.pf['pop_k_ion_igm'](z)

        if (not self.background.solve_rte[popid]) or \
            (not np.any(self.background.bands_by_pop[popid] > pop.pf['pop_EminX'])):
            
            Lx = pop.LuminosityDensity(z, Emin=pop.pf['pop_Emin_xray'], 
                Emax=pop.pf['pop_Emax'])
            
            weight = self.rate_to_coefficient(z, species, **kw)
            primary = weight * Lx \
                * (1. + z)**3 / pop.pf['pop_Ex'] / erg_per_ev
            fion = self.esec.DepositionFraction(kw['igm_e'], channel='h_1')[0]

            return primary * (1. + fion) * (pop.pf['pop_Ex'] - E_th[0]) \
                / E_th[0]

        # Full calculation - much like computing integrated flux
        norm = J21_num * self.sigma0
        
        # Integrate over function
        if kw['fluxes'][popid] is None:
            integrand = lambda E, zz: \
                self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'], 
                zxavg=kw['zxavg']) * self.sigma(E, species=species) \
                / norm / ev_per_hz
                
            ion, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0, 
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)    
        
        # Integrate over set of discrete points
        else:  
            integrand = self.sigma_E[popid][band][species] \
                * kw['fluxes'][popid][band] / norm / ev_per_hz
        
            if self.sampled_integrator == 'romb':
                ion = romb(integrand * self.E[popid][band], 
                    dx=self.dlogE[popid][band])[0] * log10
            else:
                ion = simps(integrand * self.E[popid][band], 
                    x=self.logE[popid][band]) * log10
                
        # Re-normalize
        ion *= 4. * np.pi * norm
        
        # Currently a rate coefficient, returned value depends on return_rc
        if kw['return_rc']:
            pass
        else:
            ion *= self.coefficient_to_rate(z, species, **kw) 
        
        return ion
                
    def SecondaryIonizationRateIGM(self, z, species=0, donor=0, popid=0, 
        band=0, **kwargs):
        """
        Compute volume averaged secondary ionization rate.

        Parameters
        ----------
        z : float
            redshift
        species : int
            Ionization rate of what atom?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)
        donor : int
            Which atom gave the electron?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)        

        ===============
        relevant kwargs
        ===============
        fluxes : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of 
            erg s**-1 cm**-3 atom**-1.    

        Returns
        -------
        Volume averaged ionization rate due to secondary electrons, 
        in units of ionizations per second.

        """    
        
        pop = self.pops[popid]
        
        if self.pf['secondary_ionization'] == 0:
            return 0.0

        if not pop.pf['pop_ion_src_igm']:
            return 0.0 

        # Computed in IonizationRateIGM in this case
        if not self.background.solve_rte[popid]:
            return 0.0

        if not np.any(self.background.bands_by_pop[popid] > pop.pf['pop_EminX']):
            return 0.0
        
        if ((donor or species) in [1,2]) and self.pf['approx_He']:
            return 0.0

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(**kwargs)

        #if self.pf['gamma_igm'] is not None:
        #    return self.pf['gamma_igm'](z)

        species_str = species_i_to_str[species]
        donor_str = species_i_to_str[donor]

        if self.esec.Method > 1:
            fion_const = 1.
            if kw['igm_e'] == 0:
                fion = self.fion[popid][species_str][:,0]
            else:
                i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                if self.esec.x[i_x] > kw['igm_e']:
                    i_x -= 1
                    
                j = i_x + 1    

                fion = self.fion[popid][species_str][:,i_x] \
                    + (self.fion[popid][species_str][:,j] - self.fion[popid][species_str][:,i_x]) \
                    * (kw['igm_e'] - self.esec.x[i_x]) \
                    / (self.esec.x[j] - self.esec.x[i_x])
        else:
            fion = 1.0
            fion_const = self.esec.DepositionFraction(kw['igm_e'], 
                channel=species_str)[0]

        norm = J21_num * self.sigma0
                                
        if kw['fluxes'][popid] is None:        
            if self.pf['approx_He']: # assumes lower integration limit > 4 Ryd
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'], 
                    zxavg=kw['zxavg']) * (self.sigma(E) * (E - E_th[0]) \
                    + self.cosm.y * self.sigma(E, 1) * (E - E_th[1])) \
                    / E_th[0] / norm / ev_per_hz
            else:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'], 
                    zxavg=kw['zxavg']) * self.sigma(E) * (E - E_th[0]) \
                    / E_th[0] / norm / ev_per_hz
        else:
            integrand = fion * self.sigma_E[popid][band][donor] \
                * (self.E[popid][band] - E_th[donor])
            
            if self.pf['approx_He']:
                integrand += self.cosm.y * self.sigma_E[popid][band][1] \
                    * (self.E[popid][band] - E_th[1])
            
            integrand = integrand
            integrand *= kw['fluxes'][popid][band] / E_th[species] / norm \
                / ev_per_hz
        
        if type(integrand) == types.FunctionType:
            ion, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0, 
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)
        else:
            if self.sampled_integrator == 'romb':
                ion = romb(integrand * self.E[popid][band], 
                    dx=self.dlogE[popid][band])[0] * log10
            else:
                ion = simps(integrand * self.E[popid][band], 
                    x=self.logE[popid][band]) * log10    
                
        # Re-normalize
        ion *= 4. * np.pi * norm * fion_const
                
        # Currently a rate coefficient, returned value depends on return_rc
        if kw['return_rc']:
            pass
        else:
            ion *= self.coefficient_to_rate(z, species, **kw) 
        
        return ion
        
    def DiffuseLymanAlphaFlux(self, z, **kwargs):
        """
        Flux of Lyman-alpha photons induced by photo-electron collisions.
        
        """
        
        raise NotImplemented('hey fix me')
            
        if not self.pf['secondary_lya']:
            return 0.0
        
        #return 1e-25
        
        # Grab defaults, do some patches if need be    
        kw = self._fix_kwargs(**kwargs)
                
        # Compute fraction of photo-electron energy deposited as Lya excitation
        if self.esec.Method > 1 and (kw['fluxes'][popid] is not None):
            if kw['igm_e'] == 0:
                flya = self.flya[:,0]
            else:
                i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                if self.esec.x[i_x] > kw['igm_e']:
                    i_x -= 1
                    
                j = i_x + 1    
                
                flya = self.flya[:,i_x] \
                    + (self.flya[:,j] - self.flya[:,i_x]) \
                    * (kw['igm_e'] - self.esec.x[i_x]) \
                    / (self.esec.x[j] - self.esec.x[i_x])                
        else:
            return 0.0
                
        # Re-normalize to help integrator
        norm = J21_num * self.sigma0
                
        # Compute integrand
        integrand = self.sigma_E[species] * (self.E - E_th[species])
       
        integrand *= kw['fluxes'] * flya / norm / ev_per_hz
                         
        if kw['Emax'] is not None:
            imax = np.argmin(np.abs(self.E - kw['Emax']))
            if imax == 0:
                return 0.0
                
            if self.sampled_integrator == 'romb':
                raise ValueError("Romberg's method cannot be used for integrating subintervals.")
                heat = romb(integrand[0:imax] * self.E[0:imax], dx=self.dlogE[0:imax])[0] * log10
            else:
                heat = simps(integrand[0:imax] * self.E[0:imax], x=self.logE[0:imax]) * log10
        
        else:
            imin = np.argmin(np.abs(self.E - self.pop.pf['source_Emin']))
            
            if self.sampled_integrator == 'romb':
                heat = romb(integrand[imin:] * self.E[imin:], 
                    dx=self.dlogE[imin:])[0] * log10
            elif self.sampled_integrator == 'trapz':
                heat = np.trapz(integrand[imin:] * self.E[imin:], 
                    x=self.logE[imin:]) * log10
            else:
                heat = simps(integrand[imin:] * self.E[imin:], 
                    x=self.logE[imin:]) * log10
          
        # Re-normalize, get rid of per steradian units
        heat *= 4. * np.pi * norm * erg_per_ev

        # Currently a rate coefficient, returned value depends on return_rc                                      
        if kw['return_rc']:
            pass
        else:
            heat *= self.coefficient_to_rate(z, species, **kw)

        return heat
        
    #def OpticalDepth(self, z1, z2, E, **kwargs):
    #    """
    #    Compute the optical depth between two redshifts.
    #    
    #    If no keyword arguments are supplied, assumes the IGM is neutral.
    #    
    #    Parameters
    #    ----------
    #    z1 : float
    #        observer redshift
    #    z2 : float
    #        emission redshift
    #    E : float
    #        observed photon energy (eV)  
    #        
    #    Notes
    #    -----
    #    If keyword argument 'xavg' is supplied, it must be a function of 
    #    redshift.
    #    
    #    Returns
    #    -------
    #    Optical depth between z1 and z2 at observed energy E.
    #    
    #    """
    #    
    #    #kw = self._fix_kwargs(functionify=True, **kwargs)
    #    kw = kwargs
    #    
    #    # Compute normalization factor to help numerical integrator
    #    norm = self.cosm.hubble_0 / c / self.sigma0
    #    
    #    # Temporary function to compute emission energy of observed photon
    #    Erest = lambda z: self.RestFrameEnergy(z1, E, z)
    #        
    #    # Always have hydrogen
    #    sHI = lambda z: self.sigma(Erest(z), species=0)
    #            
    #    # Figure out number densities and cross sections of everything
    #    if self.approx_He:
    #        nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #        nHeI = lambda z: nHI(z) * self.cosm.y
    #        sHeI = lambda z: self.sigma(Erest(z), species=1)
    #        nHeII = lambda z: 0.0
    #        sHeII = lambda z: 0.0
    #    elif self.self_consistent_He:
    #        if type(kw['xavg']) is not list:
    #            raise TypeError('hey! fix me')
    #            
    #        nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #        nHeI = lambda z: self.cosm.nHe(z) \
    #            * (1. - kw['xavg'](z) - kw['xavg'](z))
    #        sHeI = lambda z: self.sigma(Erest(z), species=1)
    #        nHeII = lambda z: self.cosm.nHe(z) * kw['xavg'](z)
    #        sHeII = lambda z: self.sigma(Erest(z), species=2)
    #    else:
    #        nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #        nHeI = sHeI = nHeII = sHeII = lambda z: 0.0
    #    
    #    tau_integrand = lambda z: norm * self.cosm.dldz(z) \
    #        * (nHI(z) * sHI(z) + nHeI(z) * sHeI(z) + nHeII(z) * sHeII(z))
    #        
    #    # Integrate using adaptive Gaussian quadrature
    #    tau = quad(tau_integrand, z1, z2, epsrel=self.rtol, 
    #        epsabs=self.atol, limit=self.divmax)[0] / norm
    #                                                
    #    return tau
    #
    #def Tau(self, z1, z2, E, species=0, sfrac=lambda z: 1.0):
    #    """
    #    Compute the optical depth between two redshifts.
    #
    #    If no keyword arguments are supplied, assumes the IGM is neutral.
    #
    #    Parameters
    #    ----------
    #    z1 : float
    #        observer redshift
    #    z2 : float
    #        emission redshift
    #    E : float
    #        observed photon energy (eV)  
    #    species : int
    #        Can be 0, 1, or 2 for HI, HeI, HeII
    #
    #    Notes
    #    -----
    #    If keyword argument 'xavg' is supplied, it must be a function of 
    #    redshift.
    #
    #    Returns
    #    -------
    #    Optical depth between z1 and z2 at observed energy E.
    #
    #    """
    #
    #    # Compute normalization factor to help numerical integrator
    #    norm = self.cosm.hubble_0 / c / self.sigma0
    #
    #    # Temporary function to compute emission energy of observed photon
    #    Erest = lambda z: self.RestFrameEnergy(z1, E, z)
    #
    #    # Cross-section
    #    s = lambda z: self.sigma(Erest(z), species=species)
    #    
    #    if species == 0:
    #        n = lambda z: self.cosm.nH(z) * sfrac(z)
    #    else:    
    #        n = lambda z: self.cosm.nHe(z) * sfrac(z)
    #    
    #    tau_integrand = lambda z: norm * self.cosm.dldz(z) * n(z) * s(z)
    #    
    #
    #    # Figure out number densities and cross sections of everything
    #    #if self.approx_He:
    #    #    nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #    #    nHeI = lambda z: nHI(z) * self.cosm.y
    #    #    sHeI = lambda z: self.sigma(Erest(z), species=1)
    #    #    nHeII = lambda z: 0.0
    #    #    sHeII = lambda z: 0.0
    #    #elif self.self_consistent_He:
    #    #    if type(kw['xavg']) is not list:
    #    #        raise TypeError('hey! fix me')
    #    #
    #    #    nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #    #    nHeI = lambda z: self.cosm.nHe(z) \
    #    #        * (1. - kw['xavg'](z) - kw['xavg'](z))
    #    #    sHeI = lambda z: self.sigma(Erest(z), species=1)
    #    #    nHeII = lambda z: self.cosm.nHe(z) * kw['xavg'](z)
    #    #    sHeII = lambda z: self.sigma(Erest(z), species=2)
    #    #else:
    #    #    nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
    #    #    nHeI = sHeI = nHeII = sHeII = lambda z: 0.0
    #
    #    #tau_integrand = lambda z: norm * self.cosm.dldz(z) \
    #    #    * (nHI(z) * sHI(z) + nHeI(z) * sHeI(z) + nHeII(z) * sHeII(z))
    #
    #    # Integrate using adaptive Gaussian quadrature
    #    tau = quad(tau_integrand, z1, z2, epsrel=self.rtol, 
    #        epsabs=self.atol, limit=self.divmax)[0] / norm
    #
    #    return tau
    #                    
    #