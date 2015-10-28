"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat May 23 12:13:03 CDT 2015

Description: 

"""

import inspect
import numpy as np
from ..util import read_lit
import os, pickle, inspect, re
import matplotlib.pyplot as pl
from types import FunctionType
from ..physics import Cosmology
from .Halo import HaloPopulation
from .Population import Population
from collections import namedtuple
from ..sources.Source import Source
from ..sources import Star, BlackHole
from ..util.PrintInfo import print_pop
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
from scipy.optimize import fsolve, fmin, curve_fit
from scipy.special import gamma, gammainc, gammaincc
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, erg_per_ev, rhodot_cgs, \
    E_LyA, rho_cgs, s_per_myr, cm_per_mpc, h_p, c, ev_per_hz
from ..util.SetDefaultParameterValues import StellarParameters, \
    BlackHoleParameters

try:
    from scipy.special import erfc
except ImportError:
    pass
    
try:
    import mpmath
except ImportError:
    pass    
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

ARES = os.getenv('ARES')    
log10 = np.log(10.)

def param_redshift(par):
    
    m = re.search(r"\[(\d+(\.\d*)?)\]", par)
    
    prefix = par.replace(m.group(0), '')

    return prefix, float(m.group(1))

lftypes = ['schecter', 'dpl']    
    
class LiteratureSource(Source):
    def __init__(self, **kwargs):
        self.pf = kwargs
        Source.__init__(self)
        
        _src = read_lit(self.pf['pop_sed'])
                
        if hasattr(_src, 'Spectrum'):
            if inspect.ismethod(_src.Spectrum) or \
                (type(_src.Spectrum) is FunctionType):
                self._Intensity = _src.Spectrum
            else:
                self._Intensity = _src.Spectrum(**self.pf['pop_kwargs'])
            
def normalize_sed(pop):
    """
    Convert yield to erg / g.
    """
    
    # Remove whitespace and convert everything to lower-case
    units = pop.pf['pop_yield_units'].replace(' ', '').lower()
                
    if units == 'erg/s/sfr':
        return pop.pf['pop_yield'] * s_per_yr / g_per_msun

    E1 = pop.pf['pop_EminNorm']
    E2 = pop.pf['pop_EmaxNorm']
    erg_per_phot = pop.src.AveragePhotonEnergy(E1, E2) * erg_per_ev
    
    energy_per_sfr = pop.pf['pop_yield']
    
    if units == 'photons/baryon':
        energy_per_sfr *= erg_per_phot / pop.cosm.g_per_baryon
    elif units == 'photons/msun':
        energy_per_sfr *= erg_per_phot / g_per_msun
    elif units == 'photons/s/sfr':
        energy_per_sfr *= erg_per_phot * s_per_yr / g_per_msun   
    else:
        raise ValueError('Unrecognized yield units: %s' % units)

    return energy_per_sfr

class GalaxyPopulation(HaloPopulation):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        #self.pf.update(**kwargs)
        
        self._eV_per_phot = {}
        self._conversion_factors = {}

    @property
    def is_lya_src(self):
        if not hasattr(self, '_is_lya_src'):
            self._is_lya_src = \
                self.pf['pop_Emin'] <= E_LyA <= self.pf['pop_Emax']    

        return self._is_lya_src
    
    @property
    def _Source(self):
        if not hasattr(self, '_Source_'):
            if self.pf['pop_sed'] == 'bb':
                self._Source_ = Star
            elif self.pf['pop_sed'] in ['pl', 'mcd', 'simpl']:
                self._Source_ = BlackHole
            else: 
                self._Source_ = LiteratureSource
        
        return self._Source_
        
    @property
    def src_kwargs(self):
        """
        Dictionary of kwargs to pass on to an ares.source instance.
        
        This is basically just converting pop_* parameters to source_* 
        parameters.
        
        """
        if not hasattr(self, '_src_kwargs'):
            self._src_kwargs = {}
            if self._Source is Star:
                spars = StellarParameters()
                for par in spars:
                    
                    par_pop = par.replace('source', 'pop')
                    if par_pop in self.pf:
                        self._src_kwargs[par] = self.pf[par_pop]
                    else:
                        self._src_kwargs[par] = spars[par]
                        
            elif self._Source is BlackHole:
                bpars = BlackHoleParameters()
                for par in bpars:
                    par_pop = par.replace('source', 'pop')
                    
                    if par_pop in self.pf:
                        self._src_kwargs[par] = self.pf[par_pop]
                    else:
                        self._src_kwargs[par] = bpars[par]
            else:
                self._src_kwargs = self.pf.copy()
                self._src_kwargs.update(self.pf['pop_kwargs'])
        
        return self._src_kwargs

    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = self._Source(**self.src_kwargs)
                    
        return self._src            

    @property
    def yield_per_sfr(self):
        if not hasattr(self, '_yield_per_sfr'):
            self._yield_per_sfr = normalize_sed(self)
            
        return self._yield_per_sfr
            
    @property
    def is_fcoll_model(self):
        return self.pf['pop_model'].lower() == 'fcoll'
        
    @property    
    def is_ham_model(self):
        return self.pf['pop_model'].lower() == 'ham'
    
    @property
    def is_hod_model(self):
        return self.pf['pop_model'].lower() == 'hod'
    
    @property
    def is_user_model(self):
        return self.pf['pop_model'].lower() == 'user'    
        
    @property
    def is_user_fstar(self):
        return type(self.pf['pop_fstar']) == FunctionType
        
    @property
    def rhoL_from_sfrd(self):
        if not hasattr(self, '_rhoL_from_sfrd'):
            self._rhoL_from_sfrd = self.is_fcoll_model \
                or self.is_ham_model or self.pf['pop_sfrd'] is not None
                
        return self._rhoL_from_sfrd
    
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
        
        return self._magsys
        
    @property
    def constraints(self):
        if not hasattr(self, '_constraints'):
            
            if self.pf['pop_constraints'] is not None:
                self._constraints = self.pf['pop_constraints'].copy()
            else:
                self._constraints = {}
                self._constraints['z'] = self.pf['pop_lf_z']
                self._constraints['Mstar'] = []
                self._constraints['pstar'] = []
                self._constraints['alpha'] = []

                for z in self._constraints['z']:
                    self._constraints['Mstar'].append(self.pf['pop_lf_Mstar[%g]' % z])
                    self._constraints['pstar'].append(self.pf['pop_lf_pstar[%g]' % z])
                    self._constraints['alpha'].append(self.pf['pop_lf_alpha[%g]' % z])
                            
            # Parameter file will have LF in Magnitudes...argh
            redshifts = self._constraints['z']
            self._constraints['Lstar'] = []
            
            for i, z in enumerate(redshifts):
                M = self._constraints['Mstar'][i]
                L = self.magsys.mAB_to_L(mag=M, z=z)
                self._constraints['Lstar'].append(L)
        
        return self._constraints 
    
    @property
    def Macc(self):
        """
        Mass accretion rate onto halos of mass M at redshift z.
        
        ..note:: This is the *matter* accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
        """
        if not hasattr(self, '_Macc'):
            if self.pf['pop_Macc'] is None:
                self._Macc = None
            elif type(self.pf['pop_Macc']) is FunctionType:
                self._Macc = self.pf['pop_Macc']
            else:
                self._Macc = read_lit(self.pf['pop_Macc']).Macc
            
        return self._Macc        
    
    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            # First, compute threshold mass vs. redshift
            if self.pf['pop_Mmin'] is not None:
                self._Mmin = self.pf['pop_Mmin'] * np.ones(self.halos.Nz)
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'], 
                    z, mu=self.pf['mu'])
                self._Mmin = np.array(map(Mvir, self.halos.z))

        return self._Mmin
        
    @property
    def eta(self):
        """
        Correction factor for Macc.
        
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}
        
        """
        
        if not self.is_ham_model:
            raise AttributeError('eta is a HAM thing!')

        # Prepare to compute eta
        if not hasattr(self, '_eta'):        

            self._eta = np.zeros_like(self.halos.z)

            for i, z in enumerate(self.halos.z):
                
                # eta = rhs / lhs

                Mmin = self.Mmin[i]
                
                rhs = self.cosm.rho_m_z0 * self.dfcolldt(z)
                rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3

                # Accretion onto all halos (of mass M) at this redshift
                # This is *matter*, not *baryons*
                Macc = self.Macc(z, self.halos.M)
                
                j1 = np.argmin(np.abs(Mmin - self.halos.M))
                
                if Mmin > self.halos.M[j1]:
                    j1 -= 1
                
                j2 = j1 + 1

                integ = self.halos.dndlnm[i] * Macc

                p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
                p2 = simps(integ[j2:], x=self.halos.lnM[j2:])
                
                lhs = np.interp(np.log(Mmin), self.halos.lnM[j1:j1+2],
                    [p1, p2])
                    
                self._eta[i] = rhs / lhs

        return self._eta
        
    def fstar(self, z=None, M=None):
        """
        Compute the mass- and redshift-dependent star-formation efficiency.
        """
                
        if self.is_user_fstar:
            return self.pf['pop_fstar'](z, M)
        
        elif not self.is_ham_model:
            return self.pf['pop_fstar']

        if hasattr(self, '_fstar'):
            return self._fstar_cap(self._fstar(z, M))

        self._fstar = lambda zz, MM: self._fstar_poly(zz, MM, 
            *self._fstar_coeff)
        
        fstar = self._fstar(z, M)
        return self._fstar_cap(fstar)
        
    @property
    def _Marr(self):
        if not hasattr(self, '_Marr_'):
            self._Marr_ = 10**self.pf['pop_logM']
               
        return self._Marr_

    @property    
    def _fstar_ham(self):
        """
        These are the star-formation efficiencies derived from abundance
        matching.
        """

        if hasattr(self, '_fstar_ham_pts'):
            return self._fstar_ham_pts

        # Otherwise, we're doing an abundance match!
        assert self.is_ham_model
        
        kappa_UV = self.pf['pop_kappa_UV']

        Marr = self._Marr
        
        Nz, Nm = len(self.constraints['z']), len(Marr)
        
        self._fstar_ham_pts = np.zeros([Nz, Nm])
        
        pb = ProgressBar(self._fstar_ham_pts.size, name='ham', 
            use=self.pf['progress_bar'])
        pb.start()
        
        # Do it already    
        for i, z in enumerate(self.constraints['z']):
        
            alpha = self.constraints['alpha'][i]
            L_star = self.constraints['Lstar'][i]
            phi_star = self.constraints['pstar'][i]
        
            #self.halos.MF.update(z=z)
            
            eta = self.eta[i]
            ngtm = self.halos.ngtm[np.argmin(np.abs(z - self.halos.z))]
        
            for j, M in enumerate(Marr):
        
                if M < self.Mmin[i]: 
                    continue

                # Read in variables
                Macc = self.Macc(z, M) * self.cosm.fbaryon
                
                # Minimum luminosity as a function of minimum mass
                LofM = lambda fstar: fstar * Macc * eta / kappa_UV

                # Number density of halos at masses > M
                int_nMh = np.exp(np.interp(np.log(M), self.halos.lnM, 
                    np.log(ngtm)))

                def to_min(fstar):
                    Lmin = LofM(fstar[0])
        
                    if Lmin < 0:
                        return np.inf
        
                    xmin = Lmin / L_star    
                    int_phiL = self._schecter_integral_inf(xmin, alpha)                                                      
                    int_phiL *= phi_star
        
                    return abs(int_phiL - int_nMh)
        
                fast = fsolve(to_min, 0.001, factor=0.0001, maxfev=1000)[0]
        
                self._fstar_ham_pts[i,j] = fast
        
                pb.update(i * Nm + j + 1)
        
        pb.finish()    
                        
        return self._fstar_ham_pts    
            
    @property
    def _fstar_coeff(self):
        if not hasattr(self, '_fstar_coeff_'):
            x = [self._Marr, self.constraints['z']]
            y = self._fstar_ham.flatten()
            #guess = [-160, 34., 10., -0.5, -2.5, 0.05]
            #guess = np.array([0.1, 0.1., 1., -1., -0.6, 0.01])
            guess = -1. * np.ones(6)
            
            def to_min(x, *coeff):
                M, z = np.meshgrid(*x)
                return self._fstar_poly(z, M, *coeff).flatten()
            
            try:
                self._fstar_coeff_, self._fstar_cov_ = \
                    curve_fit(to_min, x, y, p0=guess, maxfev=10000)
            except RuntimeError:
                self._fstar_coeff_, self._fstar_cov_ = \
                    guess, np.diag(guess)
                            
        return self._fstar_coeff_
        
    def _fstar_cap(self, fstar):
        if type(fstar) in [float, np.float64]:
            if fstar > self.pf['pop_fstar_ceil']:
                return self.pf['pop_fstar_ceil']
            else:
                return fstar
        elif type(fstar) == np.ndarray:
            ceil = self.pf['pop_fstar_ceil'] * np.ones_like(fstar)
            return np.minimum(fstar, ceil)    
        else:
            raise TypeError('Unrecognized type: %s' % type(fstar))
        
    def _fstar_poly(self, z, M, *coeff):
        """
        A 6 parameter model for the star-formation efficiency.
        
        References
        ----------
        Sun, G., and Furlanetto, S.R., 2015, in prep.
        
        """
                        
        logf = coeff[0] + coeff[1] * np.log10(M / 1e10) \
            + coeff[2] * ((1. + z) / 8.) \
            + coeff[3] * ((1. + z) / 8.) * np.log10(M / 1e10) \
            + coeff[4] * (np.log10(M / 1e10))**2. \
            + coeff[5] * (np.log10(M / 1e10))**3.
                
        return 10**logf
    
    @property
    def _sfr_ham(self):    
        """
        SFR as a function of redshift and halo mass yielded by abundance match.
        
            ..note:: Units are Msun/yr.
            
        """
        if not hasattr(self, '_sfr_ham_'):
            self._sfr_ham_ = np.zeros([self.halos.Nz, self.halos.Nm])
            for i, z in enumerate(self.halos.z):
                self._sfr_ham_[i] = self.cosm.fbaryon \
                    * self.Macc(z, self.halos.M) * self.fstar(z, self.halos.M)
            
        return self._sfr_ham_
    
    @property
    def _sfrd_ham(self):    
        """
        Spline fit to SFRD yielded by abundance match.
        """    
        
        if not hasattr(self, '_sfrd_ham_'):
            self._sfrd_ham_ = interp1d(self.halos.z, self._sfrd_ham_tab,
                kind='cubic')
                
        return self._sfrd_ham_
                
    @property
    def _sfrd_ham_tab(self):    
        """
        SFRD as a function of redshift yielded by abundance match.
        
            ..note:: Units are g/s/cm^3 (comoving).
            
        """
        if not hasattr(self, '_sfrd_ham_tab_'):
            self._sfrd_ham_tab_ = np.zeros(self.halos.Nz)
            
            for i, z in enumerate(self.halos.z):
                integrand = self._sfr_ham[i] * self.halos.M * self.halos.dndm[i]

                k = np.argmin(np.abs(self.Mmin[i] - self.halos.M))

                self._sfrd_ham_tab_[i] = \
                    simps(integrand[k:], x=np.log(self.halos.M[k:]))

            self._sfrd_ham_tab_ *= g_per_msun / s_per_yr / cm_per_mpc**3

        return self._sfrd_ham_tab_
        
    def _schecter_integral_inf(self, xmin, alpha):
        """
        Integral of the luminosity function from Lmin to inf (in luminosity).
    
        Parameters
        ----------
        xmin : int, float
            Lower limit of integration, (Lmin / Lstar)
        alpha : int, float
            Faint-end slope
        """

        return mpmath.gammainc(alpha + 1., xmin)
        
    #def _SchecterFunction(self, L):
    #    """
    #    Schecter function for, e.g., the galaxy luminosity function.
    #    
    #    Parameters
    #    ----------
    #    L : float
    #    
    #    """
    #    
    #    return self.phi0 * (L / self.Lstar)**self.pf['lf_slope'] \
    #        * np.exp(-L / self.Lstar)
            
    def _DoublePowerLaw(self, L):
        """
        Double power-law function for, e.g., the quasar luminosity function.
        
        Parameters
        ----------
        L : float
        
        """
        return self.pf['lf_norm'] * ((L / self.Lstar)**self.pf['lf_gamma1'] \
            + (L / self.Lstar)**self.pf['lf_gamma1'])**-1.
    
    def LuminosityFunction(self, L=None, M=None, z=None, Emin=None, Emax=None):
        """
        Compute luminosity function.

        Parameters
        ----------
        L : int, float
            Luminosity to consider
        z : int, float 
            Redshift.
        Emin : int, float
            Lower threshold of band to consider for LF [eV]
        Emax : int, float    
            Upper threshold of band to consider for LF [eV]        

        """
        
        if self.is_fcoll_model:
            raise TypeError('this is an fcoll model!')

        elif self.is_ham_model:
            # Only know LF at a few redshifts...
            assert z in self.pf['pop_lf_z']
            
            if L is None:
                
                Mst = self.pf['pop_lf_Mstar[%g]' % z]
                pst = self.pf['pop_lf_pstar[%g]' % z]
                a = self.pf['pop_lf_alpha[%g]' % z]
                
                phi_of_M = 0.4 * np.log(10) * pst \
                    * (10**(0.4 * (Mst - M)))**(1. + a) \
                    * np.exp(-10**(0.4 * (Mst - M)))
                    
                return phi_of_M
                
            else:
                raise NotImplemented('help')

        elif self.is_hod_model:
            
            self.halos.MF.update(z=z)
            dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**4
            fstar_of_m = self.fstar(z=z, M=self.halos.M)

            integrand = dndm * fstar_of_m * self.halos.M

            # Msun / cMpc**3
            integral = simps(dndm, x=self.halos.M)

            tdyn = s_per_yr * 1e6
        else:
            raise NotImplemented('need help w/ this model!')    

        L *= self._convert_band(Emin, Emax)

        if self.lf_type == 'user':
            phi = self._UserDefinedLF(L, z)
        elif self.lf_type == 'schecter':
            phi = self._SchecterFunction(L)
        elif self.lf_type == 'dpl':
            phi = self._DoublePowerLaw(L)
        else:
            raise NotImplemented('Function type %s not supported' % self.lf_type)

        return phi

    def _convert_band(self, Emin, Emax):
        """
        Convert from luminosity function in reference band to given bounds.
        
        Parameters
        ----------
        Emin : int, float
            Minimum energy [eV]
        Emax : int, float
            Maximum energy [eV]
            
        Returns
        -------
        Multiplicative factor that converts LF in reference band to that 
        defined by ``(Emin, Emax)``.
        
        """
        
        different_band = False

        # Lower bound
        if (Emin is not None) and (self.src is not None):
            different_band = True
        else:
            Emin = self.pf['pop_Emin']

        # Upper bound
        if (Emax is not None) and (self.src is not None):
            different_band = True
        else:
            Emax = self.pf['pop_Emax']
            
        # Modify band if need be
        if different_band:    
            
            if (Emin, Emax) in self._conversion_factors:
                return self._conversion_factors[(Emin, Emax)]
            
            if Emin < self.pf['pop_Emin']:
                print "WARNING: Emin < pop_Emin"
            if Emax > self.pf['pop_Emax']:
                print "WARNING: Emax > pop_Emax"    
            
            factor = quad(self.src.Spectrum, Emin, Emax)[0]
            
            self._conversion_factors[(Emin, Emax)] = factor
            
            return factor
        
        return 1.0

    def _get_energy_per_photon(self, Emin, Emax):
        # Should this go in Population
        different_band = False

        # Lower bound
        if (Emin is not None) and (self.src is not None):
            different_band = True
        else:
            Emin = self.pf['pop_Emin']

        # Upper bound
        if (Emax is not None) and (self.src is not None):
            different_band = True
        else:
            Emax = self.pf['pop_Emax']
            
        if (Emin, Emax) in self._eV_per_phot:
            return self._eV_per_phot[(Emin, Emax)]
        
        if Emin < self.pf['pop_Emin']:
            print "WARNING: Emin < pop_Emin"
        if Emax > self.pf['pop_Emax']:
            print "WARNING: Emax > pop_Emax"    
        
        integrand = lambda E: self.src.Spectrum(E) * E
        Eavg = quad(integrand, Emin, Emax)[0]
        
        self._eV_per_phot[(Emin, Emax)] = Eavg 
        
        return Eavg 

    @property
    def _sfrd(self):
        if not hasattr(self, '_sfrd_'):
            if self.pf['pop_sfrd'] is None:
                self._sfrd_ = None
            elif type(self.pf['pop_sfrd']) is FunctionType:
                self._sfrd_ = self.pf['pop_sfrd']
            elif inspect.ismethod(self.pf['pop_sfrd']):
                self._sfrd_ = self.pf['pop_sfrd']
            else:
                tmp = read_lit(self.pf['pop_sfrd'])
                self._sfrd_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self._sfrd_
    
    @property
    def _lf(self):
        if not hasattr(self, '_lf_'):
            if self.pf['pop_rhoL'] is None and self.pf['pop_lf'] is None:
                self._lf_ = None
            elif type(self.pf['pop_rhoL']) is FunctionType:
                self._lf_ = self.pf['pop_rhoL']
            elif type(self.pf['pop_lf']) is FunctionType:
                self._lf_ = self.pf['pop_lf']  
            else:
                for key in ['pop_rhoL', 'pop_lf']:
                    if self.pf[key] is None:
                        continue
                        
                    tmp = read_lit(self.pf[key])
                    self._lf_ = lambda L, z: tmp.LuminosityFunction(L, z=z,
                        **self.pf['pop_kwargs'])

                    break

        return self._lf_

    def SFRD(self, z):
        """
        Compute the comoving star formation rate density (SFRD).
    
        Given that we're in the StellarPopulation class, we are assuming
        that all emissivities are tied to the star formation history. The
        SFRD can be supplied explicitly as a function of redshift, or can 
        be computed via the "collapsed fraction" formalism. That is, compute
        the SFRD given a minimum virial temperature of star forming halos 
        (Tmin) and a star formation efficiency (fstar).
    
        If supplied as a function, the units should be Msun yr**-1 cMpc**-3.
    
        Parameters
        ----------
        z : float
            redshift
    
        Returns
        -------
        Co-moving star-formation rate density at redshift z in units of
        g s**-1 cm**-3.
    
        """
    
        if z > self.zform:
            return 0.0
        
        # SFRD approximated by some analytic function    
        if self._sfrd is not None:
            if self.pf['pop_sfrd_units'].lower() == 'g/s/cm^3':
                return self._sfrd(z)
            elif self.pf['pop_sfrd_units'].lower() == 'msun/yr/mpc^3':
                return self._sfrd(z) / rhodot_cgs
            else:
                raise NotImplemented('Unrecognized SFRD units!')
    
        # Most often: use fcoll model
        if self.is_fcoll_model:
           
            # SFRD computed via fcoll parameterization
            sfrd = self.pf['pop_fstar'] * self.cosm.rho_b_z0 * self.dfcolldt(z)
            
            if sfrd < 0:
                negative_SFRD(z, self.pf['pop_Tmin'], self.pf['pop_fstar'], 
                    self.dfcolldz(z) / self.cosm.dtdz(z), sfrd)
                sys.exit(1)
        elif self.is_ham_model:
            return float(self._sfrd_ham(z))
                
        #elif self.is_halo_model:
        #    if self.halo_model == 'hod':
        #
        #        
        #        self.halos.MF.update(z=z)
        #        dndm = self._dndm = self.halos.MF.dndm.copy() / self.cosm.h70**4
        #        fstar_of_m = self.fstar(M=self.halos.M)
        #         
        #        integrand = dndm * fstar_of_m * self.halos.M
        #        
        #        # Apply mass cut
        #        if self.pf['pop_Mmin'] is not None:
        #            iM = np.argmin(np.abs(self.halos.M - self.pf['pop_Mmin']))
        #        else:
        #            iM = 0
        #
        #        # Msun / cMpc**3
        #        integral = simps(dndm[iM:], x=self.halos.M[iM:])
        #        
        #        tdyn = s_per_myr * self.pf['pop_tSF']
        #
        #        return self.cosm.fbaryon * integral / rho_cgs / tdyn
        #        
        #    elif self.halo_model == 'clf':
        #        raise NotImplemented('havent implemented CLF yet')    
                    
        else:
            raise NotImplemented('dunno how to model the SFRD!')
    
        return sfrd                           
            
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
        
        Parameters
        ----------
        z : int, float
        
        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]
        
        """
        
        if self.rhoL_from_sfrd:
            rhoL = self.SFRD(z) * self.yield_per_sfr
        else:
            raise NotImplemented('help')    
                    
        # Convert from reference band to arbitrary band
        rhoL *= self._convert_band(Emin, Emax)
        
        if Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
            rhoL *= self.pf['pop_fesc']

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL    
    
    def NumberEmissivity(self, z, E=None, Emin=None, Emax=None):
        return self.Emissivity(z, E, Emin, Emax) / (E * erg_per_ev)

    def save(self, prefix, clobber=False):
        """
        Output population-specific data to disk.
        
        Parameters
        ----------
        prefix : str
            
        """
        
        fn = '%s.ham_coeff.pkl'
        
        if os.path.exists(fn) and (not clobber):
            raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
        
        with open(fn, 'wb') as f:
            data = (self.constraints, self._fstar_coeff)
            pickle.dump(data, f)
            
        print "Wrote %s." % fn
        

              
        