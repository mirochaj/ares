"""

GalaxyCohort.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:49:00 PST 2016

Description:

"""

import numpy as np
from inspect import ismethod
from types import FunctionType
from ..util import ProgressBar
from ..obs.Survey import Survey
from ..analysis import ModelSet
from scipy.misc import derivative
from scipy.optimize import fsolve
from ..util.Misc import numeric_types
from scipy.integrate import quad, simps, cumtrapz, ode
from scipy.interpolate import RectBivariateSpline
from .GalaxyAggregate import GalaxyAggregate
from .Population import normalize_sed
from ..util.Stats import bin_c2e, bin_e2c
from ..util.Math import central_difference, interp1d_wrapper, interp1d
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc, G, m_p, \
    k_B, h_p, erg_per_ev, ev_per_hz, sigma_T, c, t_edd, cm_per_kpc, E_LL, E_LyA, \
    cm_per_pc, m_H, s_per_myr

small_dz = 1e-8
ztol = 1e-4
#z0 = 9. # arbitrary
tiny_phi = 1e-18
#_sed_tab_attributes = ['Nion', 'Nlw', 'rad_yield', 'L1600_per_sfr',
#    'L_per_sfr', 'sps-toy']

class GalaxyCohort(GalaxyAggregate):
    """
    Create a GalaxyCohort instance.

    The defining feature of GalaxyCohort models is that galaxy properties can
    be functions of halo mass and/or redshift, providing a slight
    generalization over the GalaxyAggregate models.

    The most important parameter is `pop_sfr_model`.
    """
    #def _update_pq_registry(self, name, obj):
    #    if not hasattr(self, '_pq_registry'):
    #        self._pq_registry = {}

    #    if name in self._pq_registry:
    #        raise KeyError('{!s} already in registry!'.format(name))

    #    self._pq_registry[name] = obj

    #def __getattr__(self, name):
    #    """
    #    This gets called anytime we try to fetch an attribute that doesn't
    #    exist (yet). The only special case is really L1600_per_sfr, since
    #    that requires accessing a SynthesisModel.
    #    """

    #    # Indicates that this attribute is being accessed from within a
    #    # property. Don't want to override that behavior!
    #    # This is in general pretty dangerous but I don't have any better
    #    # ideas right now. It makes debugging hard but it's SO convenient...
    #    if (name[0] == '_'):
    #        if name.startswith('_tab'):
    #            return self.__getattribute__(name)

    #        raise AttributeError('Couldn\'t find attribute: {!s}'.format(name))

    #    # This is the name of the thing as it appears in the parameter file.
    #    full_name = 'pop_' + name

    #    # Now, possibly make an attribute
    #    try:
    #        is_php = self.pf[full_name][0:2] == 'pq'
    #    except (IndexError, TypeError):
    #        is_php = False

    #    # A few special cases
    #    _name = name
    #    if self.sed_tab and (name in _sed_tab_attributes):


    #        if name == 'rad_yield':
    #            _name = 'get_' + name
    #            att = self.src.__getattribute__(_name)
    #            val = att(self.src.Emin, self.src.Emax)
    #        else:
    #            att = self.src.__getattribute__(_name)
    #            val = att

    #        result = lambda **kwargs: val

    #    elif is_php:
    #        tmp = get_pq_pars(self.pf[full_name], self.pf)
    #        # Correct values that are strings:
    #        if self.sed_tab:
    #            pars = {}
    #            for par in tmp:
    #                if tmp[par] == 'from_sed':
    #                    pars[par] = self.src.__getattribute__(_name)
    #                else:
    #                    pars[par] = tmp[par]
    #        else:
    #            pars = tmp

    #        Mmin = lambda z: self.get_Mmin(z)
    #        #result = ParameterizedQuantity({'pop_Mmin': Mmin}, self.pf, **pars)
    #        result = ParameterizedQuantity(**pars)

    #        self._update_pq_registry(_name, result)

    #    elif type(self.pf[full_name]) in [int, float, np.int64, np.float64]:

    #        # Need to be careful here: has user-specified units!
    #        # We've assumed that this cannot be parameterized...
    #        # i.e., previous elif won't ever catch rad_yield
    #        if name == 'rad_yield':
    #            result = lambda **kwargs: normalize_sed(self)
    #        else:
    #            result = lambda **kwargs: self.pf[full_name]

    #    elif type(self.pf[full_name]) is FunctionType:
    #        result = lambda **kwargs: self.pf[full_name](**kwargs)
    #    else:
    #        raise TypeError('dunno how to handle: {!s}'.format(name))

    #    # Check to see if Z?
    #    setattr(self, name, result)

    #    return result

    def _get_src_by_Z(self, i):
        if not hasattr(self, '_src_by_Z'):
            self._src_by_Z = {}

        if i in self._src_by_Z:
            return self._src_by_Z[i]

        Zarr = np.sort(list(self.src.tab_metallicities))
        kw = self.src_kwargs.copy()
        kw['source_Z'] = Zarr[i]
        src = self._Source(cosm=self.cosm, **kw)
        self._src_by_Z[i] = src
        return src

    def _get_lum_all_Z(self, x=1600., band=None, window=1, raw=False,
        nebular_only=False, age=None, units='Angstrom'):
        """
        Get the luminosity (per SFR) for all possible metallicities.

        Returns
        -------
        A function that takes in log10(metallicity) and returns
        log10(luminosity in erg/s/Hz/(Msun/yr)).
        """

        if not hasattr(self, '_cache_lum_all_Z'):
            self._cache_lum_all_Z = {}

        # Grab result from cache if it exists.
        if (x, window, band, raw, nebular_only, age) in self._cache_lum_all_Z:
            L_of_Z_func = \
                self._cache_lum_all_Z[(x, window, band, raw, nebular_only, age)]

            return L_of_Z_func

        tmp = []
        Zarr = np.sort(list(self.src.tab_metallicities))
        for i, Z in enumerate(Zarr):
            src = self._get_src_by_Z(i)
            L_per_sfr = src.get_lum_per_sfr(x=x, units=units, window=window,
                band=band, raw=raw, nebular_only=nebular_only, age=age)

            tmp.append(L_per_sfr)

        # Interpolant
        L_of_Z_func = interp1d_wrapper(np.log10(Zarr), np.log10(tmp),
            self.pf['interp_Z'])

        self._cache_lum_all_Z[(x, window, band, raw, nebular_only, age)] = \
            L_of_Z_func

        return L_of_Z_func

    def get_radiative_yield(self, z, Mh):
        """
        Returns the total output [in erg/s] in the band defined by
        `pop_EminNorm`, `pop_EmaxNorm`. Or, if this population's SED is from
        a stellar population synthesis model (or other lookup table), it
        is the total output in the energy range covered in that table.
        """

        if self.pf['pop_rad_yield'] is not None:
            pass

    def get_metallicity(self, z, Mh=None, gas_phase=False):
        """
        Get the metallicity of all halos in the model.

        ..note :: This is a derived quantity, which is why it's not accessible
            via `get_field`.

        Returns
        -------
        Metallicity of all halos in the model, Z.

        """

        if not self.pf['pop_enrichment']:
            return np.zeros_like(self.halos.tab_M)

        assert self.pf['pop_enrichment'] == 1, \
            "Only pop_enrichment=1 available for GalaxyCohort right now."

        fb = self.cosm.fbaryon
        fmr = self.pf['pop_mass_yield']
        fZy = fmr * self.pf['pop_metal_yield']

        if Mh is None:
            Mh = self.halos.tab_M

        if self.is_user_smhm:

            smhm = self.get_smhm(z=z, Mh=Mh)
            smhm[Mh < self.get_Mmin(z)] = 0
            smhm[Mh > self.get_Mmax(z)] = 0
            Ms = smhm * Mh

            if self.pf['pop_mzr'] is not None:
                func = self._get_function('pop_mzr')

                Zgas = func(z=z, Ms=np.log10(Ms))

                if gas_phase:
                    Z = Zgas
                else:
                    Z = 10**(Zgas - 12.) \
                        * self.cosm.X / self.pf['pop_fox']

            else:
                MZ = Ms * fZy * (1. - self.get_metal_loss(z, Mh=Mh))

                # Cosmological gas mass minus stellar mass
                Mg0 = Mh * fb * (1. - smhm)

                Mg = Mg0 * (1. - self.get_mass_loss(z, Mh=Mh))

                Zstell = MZ / Mg

                if gas_phase:
                    Z = 12. + np.log10(self.pf['pop_fox'] * Zstell / self.cosm.X)
                else:
                    Z = Zstell

        else:

            Ms = self.get_field(z, 'Ms')
            MZ = Ms * fZy
            Mg = self.get_field(z, 'Mg')

            Z = MZ / Mg / self.pf['pop_fpoll']
            #Z[Mg==0] = 1e-3

            assert not gas_phase

        # Enforce metallicity floor
        #if not gas_phase:
        #    Z = np.maximum(Z, 1e-3)

        if Mh is None:
            return Z
        else:
            _Mh = self.get_field(z, 'Mh')
            return 10**np.interp(np.log10(Mh), np.log10(_Mh), np.log10(Z))

    def get_metal_loss(self, z, Mh=None):
        func = self._get_function('pop_metal_loss')
        return func(z=z, Mh=Mh)

    def get_mass_loss(self, z, Mh=None):
        func = self._get_function('pop_mass_loss')
        return func(z=z, Mh=Mh)

    def get_gas_mass(self, z, Mh=None):
        """
        Uses `pop_gas_fraction`, which is defined relative to cosmic baryon
        fraction, so a value of unity means a halo has Mgas = Mh * fbaryon.

        .. note :: This does not subtract off mass in stars, so Mgas+Mstell
            could in principle exceed fbaryon * Mh.

        """
        func = self._get_function('pop_gas_fraction')
        return func(z=z, Mh=Mh) * Mh * self.cosm.fbaryon

    def get_size(self, z, Ms):
        func = self._get_function('pop_msr')
        return func(z=z, Ms=Ms)

    def get_field(self, z, field):
        """
        Return results from SAM (all masses) at input redshift.

        .. note :: This is really similar to `get_mass`, the only
            difference is this routine uses the raw field names rather than,
            e.g., kind='stellar' for 'Mst', etc. It also doesn't do any
            interpolation onto a supplied Mh grid -- all returned values are
            their "raw" values straight out of the SAM.

        Parameters
        ----------
        z : int, float
            Redshift
        field : str
            For example, 'Mh', 'Mst', 'MZ', etc.

        Returns
        -------
        Array of field values for all halos at redshift `z`.

        """
        if self.is_user_smhm:
            if field == 'Mh':
                return self.halos.tab_M
            elif field == 'Ms':
                smhm = self.get_smhm(z=z, Mh=self.halos.tab_M)
                mste = self.halos.tab_M * smhm
                return mste
            else:
                raise NotImplementedError('help')
        else:
            zall, data = self.get_histories()
            iz = np.argmin(np.abs(z - zall))

            return data[field][:,iz]

    def get_photons_per_Msun(self, Emin, Emax):
        """
        Compute photon luminosity in band of interest per unit SFR for
        all halos.

        Returns
        -------
        In units of photons/Msun.

        """
        if not hasattr(self, '_N_per_Msun'):
            self._N_per_Msun = {}

        # If we've already figured it out, just return
        if (Emin, Emax) in self._N_per_Msun:
            return self._N_per_Msun[(Emin, Emax)]

        # Otherwise, calculate what it should be
        if (Emin, Emax) in [(E_LL, 24.6), (13.6, 24.6)]:
            # Should be based on energy at this point, not photon number
            self._N_per_Msun[(Emin, Emax)] = self.get_Nion(z=None, Mh=self.halos.tab_M) \
                * self.cosm.b_per_msun
        elif (Emin, Emax) == (10.2, 13.6):
            self._N_per_Msun[(Emin, Emax)] = self.get_Nlw(z=None, Mh=self.halos.tab_M) \
                * self.cosm.b_per_msun
        else:
            self._N_per_Msun[(Emin, Emax)] = self.get_Nlw(z=None, Mh=self.halos.tab_M) \
                * self.cosm.b_per_msun
            #s = 'Unrecognized band: ({0:.3g}, {1:.3g})'.format(Emin, Emax)
            #raise NotImplementedError(s)

        return self._N_per_Msun[(Emin, Emax)]

    @property
    def _func_nh(self):
        if not hasattr(self, '_func_nh_'):
            self._func_nh_ = \
                RectBivariateSpline(self.halos.tab_z, np.log(self.halos.tab_M),
                    self.halos.tab_dndm)
        return self._func_nh_

    @property
    def _tab_MAR_at_Mmin(self):
        if not hasattr(self, '_tab_MAR_at_Mmin_'):
            self._tab_MAR_at_Mmin_ = \
                np.array([self.get_mar(self.halos.tab_z[i], self._tab_Mmin[i]) \
                    for i in range(self.halos.tab_z.size)])

        return self._tab_MAR_at_Mmin_

    def _get_luminosity_density(self, Emin=None, Emax=None):
        """
        Compute the luminosity density in some bandpass for all redshifts.

        This is the most general way of computing the luminosity density as it
        takes into account all (existing) Mh- and z-dependent quantities.

        Returns
        -------
        Interpolant for luminosity density in units of erg / s / (comoving cm)**3.
        """

        if not hasattr(self, '_rho_L'):
            self._rho_L = {}

        # If we've already figured it out, just return
        _Emin = round(Emin, 1)
        _Emax = round(Emax, 1)
        if (_Emin, _Emax) in self._rho_L:
            return self._rho_L[(_Emin, _Emax)]

        # If nothing is supplied, compute the "full" luminosity density
        if (Emin is None) and (Emax is None):
            Emin = self.pf['pop_EminNorm']
            Emax = self.pf['pop_EmaxNorm']
        # Make sure we don't emit in bands...where we shouldn't be emitting
        elif (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                self._rho_L[(Emin, Emax)] = lambda z: 0.0
                return self._rho_L[(Emin, Emax)]
            if (Emax < self.pf['pop_Emin']):
                self._rho_L[(Emin, Emax)] = lambda z: 0.0
                return self._rho_L[(Emin, Emax)]
        else:
            raise ValueError('help!')

        need_sam = False
        use_yield_per_sfr = True

        # For all halos. Reduce to a function of redshift only by passing
        # in the array of halo masses stored in 'halos' attribute.
        if Emax <= self.pf['pop_Emin_xray']:
            N_per_Msun = self.get_photons_per_Msun(Emin=Emin, Emax=Emax)

            # Also need energy per photon in this case
            erg_per_phot = self.src.erg_per_phot(Emin, Emax)

            # Get an array for fesc

            if Emin in [13.6, E_LL]:
                # Doesn't matter what Emax is
                fesc = lambda **kwargs: self.get_fesc_UV(**kwargs)
            elif (Emin, Emax) in [(10.2, 13.6), (E_LyA, E_LL)]:
                fesc = lambda **kwargs: self.get_fesc_LW(**kwargs)
            else:
                use_yield_per_sfr = False
                fesc = lambda **kwargs: 1.0
                #return None

            yield_per_sfr = lambda **kwargs: fesc(**kwargs) \
                * N_per_Msun * erg_per_phot

            # For de-bugging purposes
            #if not hasattr(self, '_yield_by_band'):
            #    self._yield_by_band = {}
            #    self._fesc_by_band = {}

            #if (Emin, Emax) not in self._yield_by_band:
            #    self._yield_by_band[(Emin, Emax)] = yield_per_sfr
            #    self._fesc_by_band[(Emin, Emax)] = fesc

        else:
            # X-rays separate because we never have lookup table.
            # could change in the future.

            try:
                if self.rad_yield.func_var not in ['z', 'Mh']:
                    need_sam = True
            except AttributeError:
                pass

            if need_sam:
                sam_data = self.histories
                sam_z = self.histories['z']
            else:
                pass

            yield_per_sfr = lambda **kwargs: self.get_radiative_yield(**kwargs) \
                * s_per_yr

        ##
        # Now that we have yield/SFR, go through and determine SFR over
        # entire halo population to determine full luminosity density.
        ok = ~self._tab_sfr_mask
        tab = np.zeros(self.halos.tab_z.size)
        for i, z in enumerate(self.halos.tab_z):
            if z > self.zform:
                continue

            # Must grab stuff vs. Mh and interpolate to self.halos.tab_M
            # They are guaranteed to have the same redshifts.
            if need_sam:

                kw = {'z': z, 'Mh': self.halos.tab_M}
                if self.is_sfe_constant:
                    for key in sam_data.keys():
                        if key == 'Mh':
                            continue

                        kw[key] = np.interp(self.halos.tab_M,
                            sam_data['Mh'][-1::-1], sam_data[key][-1::-1])
                else:
                    raise NotImplemented('help')

            else:
                kw = {'z': z, 'Mh': self.halos.tab_M}


            if use_yield_per_sfr:
                integrand = self.tab_sfr[i] * self.halos.tab_dndlnm[i] \
                    * self.tab_focc[i] * yield_per_sfr(**kw) * ok[i]
            else:
                print("is this ever called")
                # [erg/s]
                lum_v_Mh = self.get_lum(z, band=(Emin, Emax),
                    band_units='eV')

                integrand = lum_v_Mh * self.halos.tab_dndlnm[i] \
                    * self.tab_focc[i] * ok[i]


            _tot = np.trapz(integrand, x=np.log(self.halos.tab_M))
            _cumtot = cumtrapz(integrand, x=np.log(self.halos.tab_M),
                initial=0.0)

            _tmp = _tot - \
                np.interp(np.log(self._tab_Mmin[i]), np.log(self.halos.tab_M),
                    _cumtot)

            tab[i] = _tmp

        tab *= 1. / s_per_yr #/ cm_per_mpc**3

        self._rho_L[(_Emin, _Emax)] = interp1d(self.halos.tab_z, tab,
            kind=self.pf['pop_interp_sfrd'])

        return self._rho_L[(_Emin, _Emax)]

    #def rho_N(self, z, Emin, Emax):
    #    return self._get_photon_density(z, Emin, Emax)

    def _get_photon_density(self, z, Emin, Emax):
        """
        Compute the photon luminosity density in some bandpass at some redshift.

        Returns
        -------
        Luminosity density in units of photons / s / cMpc**3.
        """

        if not hasattr(self, '_rho_N'):
            self._rho_N = {}

        # If we've already figured it out, just return
        if (Emin, Emax) in self._rho_N:
            return self._rho_N[(Emin, Emax)](z)

        tab = np.ones_like(self.halos.tab_z)

        # For all halos
        N_per_Msun = self.get_photons_per_Msun(Emin=Emin, Emax=Emax)

        if abs(Emin - E_LL) < 0.1:
            fesc = self.get_fesc(z=z, Mh=self.halos.tab_M)
        elif (abs(Emin - E_LyA) < 0.1 and abs(Emax - E_LL) < 0.1):
            fesc = self.get_fesc_LW(z=z, Mh=self.halos.tab_M)
        else:
            raise NotImplementedError('help!')

        ok = ~self._tab_sfr_mask
        for i, z in enumerate(self.halos.tab_z):
            integrand = self.tab_sfr[i] * self.halos.tab_dndlnm[i] \
                * self.tab_focc[i] * N_per_Msun * fesc * ok[i]

            tot = np.trapz(integrand, x=np.log(self.halos.tab_M))
            cumtot = cumtrapz(integrand, x=np.log(self.halos.tab_M),
                initial=0.0)

            tab[i] = tot - \
                np.interp(np.log(self._tab_Mmin[i]), np.log(self.halos.tab_M), cumtot)

        tab *= 1. / s_per_yr #/ cm_per_mpc**3

        self._rho_N[(Emin, Emax)] = interp1d(self.halos.tab_z, tab,
            kind=self.pf['pop_interp_sfrd'])

        return self._rho_N[(Emin, Emax)](z)

    def get_focc(self, z, Mh):
        """
        Get occupation fraction.
        """
        func = self._get_function('pop_focc')
        result = func(z=z, Mh=Mh)
        return result

    def get_fsurv(self, z, Mh):
        """
        Get survival fraction.
        """
        func = self._get_function('pop_fsurv')
        result = func(z=z, Mh=Mh)
        return result

    def get_fshock(self, **kwargs):
        """
        Get survival fraction.
        """
        func = self._get_function('pop_fshock')
        result = func(**kwargs)
        return result

    def get_sfrd(self, z):
        """
        Compute star-formation rate density (SFRD) at redshift `z`.
        """

        if not hasattr(self, '_func_sfrd'):
            func = interp1d(self.halos.tab_z, self.tab_sfrd_total,
                kind=self.pf['pop_interp_sfrd'])
            self._func_sfrd = func

        return self._func_sfrd(z)

    def get_smd(self, z):
        """
        Compute stellar mass density (SMD) at redshift `z`.

        Returns
        -------
        Stellar mass density in Msun/cMpc^3.

        """

        if not hasattr(self, '_func_smd'):
            if self.is_quiescent:
                assert self.pf['pop_sfr_model'] == 'smhm-func'

                self._tab_smd = np.zeros_like(self.halos.tab_z)
                for i, _z in enumerate(self.halos.tab_z):
                    smhm = self.get_smhm(z=_z, Mh=self.halos.tab_M)
                    smhm[self.halos.tab_M < self.get_Mmin(_z)] = 0
                    smhm[self.halos.tab_M > self.get_Mmax(_z)] = 0

                    integ = smhm * self.halos.tab_M * self.halos.tab_dndlnm[i] \
                        * self.tab_focc[i]
                    self._tab_smd[i] = np.trapz(integ, x=np.log(self.halos.tab_M))
            else:
                dtdz = np.array([self.cosm.dtdz(z) \
                    for z in self.halos.tab_z])
                self._tab_smd = cumtrapz(self.tab_sfrd_total[-1::-1] * dtdz[-1::-1],
                    dx=np.abs(np.diff(self.halos.tab_z[-1::-1])), initial=0.)[-1::-1]

            #self._func_smd = interp1d(self.halos.tab_z, self._tab_smd,
            #    kind=self.pf['pop_interp_sfrd'], left=0, right=0)
            self._func_smd = lambda zz: 10**np.interp(zz, self.halos.tab_z,
                np.log10(self._tab_smd))

        return self._func_smd(z)

    def get_mar(self, z, Mh):
        MGR = np.maximum(self.MGR(z, Mh), 0.)
        eta = self.eta(z, Mh)
        return eta * MGR

    @property
    def eta(self):
        if not hasattr(self, '_eta'):
            if np.all(self._tab_eta == 1):
                self._eta = lambda z, Mh=None: 1.
            elif self._tab_eta.ndim == 1:
                self._eta = lambda z, Mh=None: np.interp(z, self.halos.tab_z, self._tab_eta)
            else:
                _eta = RectBivariateSpline(self.halos.tab_z,
                    np.log10(self.halos.tab_M), self._tab_eta)
                self._eta = lambda z, Mh: _eta(z, np.log10(Mh)).squeeze()

        return self._eta

    @property
    def _tab_eta(self):
        """
        Correction factor for MAR.

        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}

        """

        # Prepare to compute eta
        if not hasattr(self, '_tab_eta_'):

            if self.pf['pop_MAR_corr'] == 'integral':

                _rhs = np.zeros_like(self.halos.tab_z)
                _lhs = np.zeros_like(self.halos.tab_z)
                _tab_eta_ = np.ones_like(self.halos.tab_z)

                for i, z in enumerate(self.halos.tab_z):

                    # eta = rhs / lhs

                    Mmin = self._tab_Mmin[i]

                    # My Eq. 3
                    rhs = self.cosm.rho_cdm_z0 * self.dfcolldt(z)
                    rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3

                    # Accretion onto all halos (of mass M) at this redshift
                    # This is *matter*, not *baryons*
                    MAR = self.halos.tab_MAR[i]

                    # Find Mmin in self.halos.tab_M
                    j1 = np.argmin(np.abs(Mmin - self.halos.tab_M))
                    if Mmin > self.halos.tab_M[j1]:
                        j1 -= 1

                    integ = self.halos.tab_dndlnm[i] * MAR

                    p0 = simps(integ[j1-1:], x=np.log(self.halos.tab_M)[j1-1:])
                    p1 = simps(integ[j1:], x=np.log(self.halos.tab_M)[j1:])
                    p2 = simps(integ[j1+1:], x=np.log(self.halos.tab_M)[j1+1:])
                    p3 = simps(integ[j1+2:], x=np.log(self.halos.tab_M)[j1+2:])

                    interp = interp1d(np.log(self.halos.tab_M)[j1-1:j1+3], [p0,p1,p2,p3],
                        kind=self.pf['pop_interp_MAR'])

                    lhs = interp(np.log(Mmin))

                    _lhs[i] = lhs
                    _rhs[i] = rhs

                    _tab_eta_[i] = rhs / lhs

                # Re-shape to be (z, Mh)
                self._tab_eta_ = np.reshape(np.tile(_tab_eta_, self.halos.tab_M.size),
                    (self.halos.tab_z.size, self.halos.tab_M.size))

                ok = np.isfinite(self._tab_eta)
                self._tab_eta[~ok] = 0

            elif self.pf['pop_MAR_corr'] == 'slope':

                # In this case, we have the same shape as _tab_MAR
                self._tab_eta_ = np.ones_like(self.halos.tab_MAR)

                # Compute power-law slope as function of mass at all redshifts.
                # Prevent dramatic deviations from this slope, and instead
                # extrapolate from high-mass end downward.
                logM = np.log(self.halos.tab_M)
                logMAR = np.log(self.halos.tab_MAR)

                alpha = np.diff(logMAR, axis=1) / np.diff(logM)
                self._tab_alpha = alpha

                Ms = self.halos.tab_M.size - 1

                # Figure out where alpha < 0, use slope well above this
                # juncture to extrapolate.
                negative = np.zeros_like(self.halos.tab_z)
                for i in range(self.halos.tab_z.size):
                    huge = np.argwhere(alpha[i,:-5] > 2.)
                    if not np.any(huge):
                        continue

                    ilt0 = int(huge[-1])

                    # Extrapolate
                    _Msafe = min(self.halos.tab_M[ilt0] * 10, 1e13)
                    iM = np.argmin(np.abs(_Msafe - self.halos.tab_M))
                    Msafe = self.halos.tab_M[iM]

                    new = self.halos.tab_MAR[i,iM] \
                        * (self.halos.tab_M / Msafe)**alpha[i,iM]

                    # Only apply extrapolation at low mass
                    self._tab_eta_[i,0:iM] = \
                        new[0:iM] / self.halos.tab_MAR[i,0:iM]


            else:
                self._tab_eta_ = np.ones_like(self.halos.tab_dndm)

        return self._tab_eta_

    def get_sfr(self, z, Mh=None):
        """
        Get star formation rate at redshift z in a halo of mass Mh.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        Mh : optional float, np.ndarray
            Halo masses [Msun]. If not suplied, we'll look to see if user
            supplied halo catalog of the form (M, x, y, z) in `pop_halos`.
            Otherwise, will default to `self.halos.tab_M`.

        P.S. When you plot this, don't freak out if the slope changes at Mmin.
        It's because all masses below this (and above Mmax) are just set to
        zero, so it looks like a slope change for line plots since it's
        trying to connect to a point at SFR=Mh=0.

        """

        # User may have supplied a function for SFR(z, Mh) directly.
        #if self.pf['pop_sfr'] is not None:
        #    func = self._get_function('pop_sfr')
        #    if type(self.pf['pop_sfr']) == 'str':
        #        return func(z=z, Mh=Mh)

        # Will use only if interpolating onto user-supplied set of halo masses.
        flipM = False

        # If Mh is None, it triggers use of tab_sfr, which spans all
        # halo masses in self.halos.tab_M

        # User may have supplied halo catalog, in which case we'll interpolate
        # onto self.halos.tab_M momentarily.
        if self.pf['pop_halos'] is not None:
            Mh, _x, _y, _z = self.pf['pop_halos'](z=z).T

            if not np.all(np.diff(Mh) > 0):
                Mh = Mh[-1::-1]
                flipM = True

        elif Mh is None:
            k = np.argmin(np.abs(z - self.halos.tab_z))
            if abs(z - self.halos.tab_z[k]) < ztol:
                return self.tab_sfr[k] #* ~self._tab_sfr_mask[k]
            else:
                Mh = self.halos.tab_M

        # Create interpolant to be self-consistent
        # with _tab_sfr. Note that this is slower than it needs to be
        # in cases where we just want to know the SFR at a few redshifts
        # and/or halo masses. But, we're rarely doing such things.
        if not hasattr(self, '_spline_sfr'):
            log10sfr = np.log10(self.tab_sfr)# * ~self._tab_sfr_mask)
            # Filter zeros since we're going log10
            log10sfr[np.isinf(log10sfr)] = -90.
            log10sfr[np.isnan(log10sfr)] = -90.

            _spline_sfr = RectBivariateSpline(self.halos.tab_z,
                np.log10(self.halos.tab_M), 10**log10sfr)

            #func = lambda z, log10M: 10**_spline_sfr(z, log10M).squeeze()

            def func(zz, log10M):
                sfr = _spline_sfr(zz, log10M).squeeze()

                M = 10**log10M

                # Check for ndim == 0 is for fringe case where
                # there's a single halo.
                if type(sfr) == np.ndarray:

                    if sfr.ndim == 0:
                        sfr = np.array([float(sfr)])

                    sfr[M < self.Mmin(zz)] = 0.0
                    sfr[M > self.get_Mmax(zz)] = 0.0
                else:
                    if M < self.Mmin(z):
                        return 0.0
                    if M > self.get_Mmax(z):
                        return 0.0

                return sfr

            self._spline_sfr = func


        # Actually evaluate SFR
        sfr = self._spline_sfr(z, np.log10(Mh))

        if flipM:
            return sfr[-1::-1]
        else:
            return sfr

        #return self.cosm.fbar_over_fcdm * self.get_MAR(z, Mh) * self.eta(z) \
        #    * self.SFE(z=z, Mh=Mh)

    def get_zindex(self, z):
        """
        Get index such that user-supplied `z` will lie in range given by:
            (self.halos.tab_z[iz], self.halos.tab_z[iz+1])

        """

        if z < self.halos.tab_z.min():
            raise ValueError(f"z={z} < tabulated range!")
        if z > self.halos.tab_z.max():
            raise ValueError(f"z={z} > tabulated range!")

        iz = np.argmin(np.abs(z - self.halos.tab_z))

        if z > self.halos.tab_z[iz]:
            iz -= 1

        return iz

    def get_emissivity(self, z, x=None, band=None, units='eV'):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].

        Parameters
        ----------
        z : int, float

        Returns
        -------
        Emissivity in units of erg / s / cMpc**3 [/ eV]

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        # Use GalaxyAggregate's Emissivity function
        if self.is_emissivity_scalable:
            # The advantage here is that the SFRD only has to be calculated
            # once, and the radiation field strength can just be determined
            # by scaling the SFRD.
            rhoL = super(GalaxyCohort, self).get_emissivity(z, x=x,
                band=band, units=units)
        else:
            iz = self.get_zindex(z)
            z1 = self.halos.tab_z[iz]
            z2 = self.halos.tab_z[iz+1]

            L1 = self.get_lum(z1, x=x, band=band, units='eV')
            L2 = self.get_lum(z2, x=x, band=band, units='eV')

            ok1 = np.logical_and(self.halos.tab_M >= self.get_Mmin(z1),
                self.halos.tab_M < self.get_Mmax(z1))
            ok2 = np.logical_and(self.halos.tab_M >= self.get_Mmin(z1),
                self.halos.tab_M < self.get_Mmax(z2))

            integ1 = L1 * self.halos.tab_dndlnm[iz,:] * self.tab_focc[iz,:]
            integ2 = L2 * self.halos.tab_dndlnm[iz+1,:] * self.tab_focc[iz+1,:]

            rhoL1 = np.trapz(integ1 * ok1, x=np.log(self.halos.tab_M))
            rhoL2 = np.trapz(integ2 * ok2, x=np.log(self.halos.tab_M))

            rhoL = rhoL1 + (z - z1) * (rhoL2 - rhoL1) / (z2 - z1)

        #else:
        #    # Here, the radiation backgrounds cannot just be scaled.
        #    # Note that this method can always be used, it's just less
        #    # efficient because you're basically calculating the SFRD again
        #    # and again.
        #    rhoL = self._get_luminosity_density(Emin, Emax)(z)

        if x is not None:
            return rhoL * self.src.get_spectrum(x, units=units) * on
        else:
            return rhoL * on

    def get_mass(self, z, Mh=None, kind='halo'):
        """
        Return the mass in some galaxy 'phase' for a given halo at redshift `z`.

        .. note :: By default, if Mh is not supplied we'll take it to be the
            halo masses at which the HMF is tabulated, self.halos.tab_M. In
            general, our halos are NOT at these masses, since they are evolved
            in time according to their MAR, SFE, etc. As a result, Mh=None
            necessarily results in interpolation of different galaxy masses
            onto the Mh values in self.halos.tab_M.

        Parameters
        ----------
        z : int, float
            Redshift
        Mh : int, float
            Halo mass [Msun]
        kind : str
            Phase of interest, e.g., 'stellar', 'metal', 'gas'.

        Returns
        -------
        Mass in Msun of desired galaxy phase.

        """
        zall, data = self.get_histories()
        iz = np.argmin(np.abs(z - zall))

        if kind in ['halo']:
            return data['Mh'][:,iz]

        if self.pf['pop_halos'] is not None:
            Mh, x, y, z = self.pf['pop_halos'](z=z).T
        elif Mh is None:
            Mh = self.halos.tab_M

        if kind in ['stellar', 'stars']:
            return np.interp(Mh, data['Mh'][:,iz], data['Ms'][:,iz])
        elif kind in ['stellar_cumulative', 'stars_cumulative']:
            return np.interp(Mh, data['Mh'][:,iz], data['Ms'][:,iz])
        elif kind in ['metal', 'metals']:
            return np.interp(Mh, data['Mh'][:,iz], data['MZ'][:,iz])
        elif kind in ['gas']:
            return np.interp(Mh, data['Mh'][:,iz], data['Mg'][:,iz])
        else:
            raise NotImplementedError('Unrecognized mass kind={}.'.format(kind))

    def get_smf(self, z, bins=None, units='dex'):
        return self.get_mf(z, bins=bins, units=units, mass='stellar')

    def get_mf(self, z, bins=None, units='dex', mass='stellar'):
        """
        Return stellar mass function.

        Parameters
        ----------
        z : int, float
            Redshift.
        bins : list, np.ndarray
            Array of log10(stellar mass / Msun) centers.

        Returns
        -------
        Tuple containing (bin centers, stellar mass function).
        """

        ##
        # Can access directly for SMHM-based parameterization.
        if self.is_user_smhm:
            iz = np.argmin(np.abs(z - self.halos.tab_z))
            dndlogm = self.halos.tab_dndlnm[iz,:] * np.log(10.)

            logMh = np.log10(self.halos.tab_M)
            logMh_e = bin_c2e(logMh)

            if mass == 'stellar':
                fstar = self.tab_fstar[iz,:]
                Ms_c = fstar * self.halos.tab_M

                fstar_e = self.get_sfe(z=z, Mh=10**logMh_e)
                Ms_e = fstar_e * 10**logMh_e
                logMs_e = np.log10(Ms_e)

                dlogmdlogM = np.diff(logMh_e) / np.diff(logMs_e)

                logMc = np.log10(Ms_c)

            elif mass == 'gas':
                Mg_c = self.get_gas_mass(z=z, Mh=self.halos.tab_M)
                Mg_e = self.get_gas_mass(z=z, Mh=10**logMh_e)

                dlogmdlogM = np.diff(logMh_e) / np.diff(np.log10(Mg_e))

                logMc = np.log10(Mg_c)
            else:
                raise NotImplementedError('help')



            phi = dndlogm * dlogmdlogM * self.tab_focc[iz,:]

            if bins is not None:
                return bins, np.interp(bins, logMc, phi)
            else:
                return logMc, phi

        ##
        # Otherwise, we integrate trajectories.
        zall, traj_all = self.get_histories()
        iz = np.argmin(np.abs(z - zall))
        Ms = traj_all['Ms'][:,iz]
        Mh = traj_all['Mh'][:,iz]
        nh = traj_all['nh'][:,iz]

        if bins is None:
            bin = 0.1
            bin_e = np.arange(6., 13.+bin, bin)
        else:
            dx = np.diff(bins)
            assert np.all(np.diff(dx) == 0)
            bin = dx[0]
            bin_e = bins

        bin_c = bin_e2c(bin_e)

        phi, _bins = np.histogram(Ms, bins=10**bin_e, weights=nh)

        if units == 'dex':
            # Convert to dex**-1 units
            phi /= bin
        else:
            raise NotImplemented('help')

        if bins is None:
            return 10**bin_c, phi
        else:
            return phi

    def get_surface_density(self, z, maglim=None, dz=1., dtheta=1., x=1600.,
        units='Angstroms', window=1):
        """
        Get the surface density of galaxies in a given redshift chunk.

        Parameters
        ----------
        z : int, float
            Redshift of galaxy population.
        maglim : int, float
            Apparent AB magnitude defining cut.
        dz : int, float
            Thickness of redshift chunk.
        dtheta : int, float
            Angle of field of view. Default: 1 deg^2.

        Returns
        -------
        Observed magnitudes, then, projected surface density of galaxies in
        `dz` thick shell, in units of cumulative number of galaxies per
        square degree. Will return as function of apparent AB magnitude, or
        if `maglim` is supplied, just the number density brighter than that
        cut.

        """

        # These are intrinsic (i.e., not dust-corrected) absolute magnitudes
        bins = np.arange(0, 40, 0.1)
        mags, phi = self.get_lf(z, bins, x=x, units=units, window=window,
            use_mags=True, absolute=False)

        # Compute the volume of the shell we're looking at
        vol = self.cosm.ProjectedVolume(z, angle=dtheta, dz=dz)

        Ngal = phi * vol

        # Cumulative surface density of galaxies *brighter than* Mobs
        cgal = cumtrapz(Ngal, x=mags, initial=Ngal[0])

        if maglim is not None:
            return np.interp(maglim, mags, cgal)
        else:
            return mags, cgal

    @property
    def is_uvlf_parametric(self):
        if not hasattr(self, '_is_uvlf_parametric'):
            self._is_uvlf_parametric = self.pf['pop_uvlf'] is not None
        return self._is_uvlf_parametric

    def _get_uvlf_mags(self, z, bins, x=1600., units='Angstroms',
        window=1, absolute=True):

        if self.is_uvlf_parametric:
            assert absolute
            func = self._get_function('pop_uvlf')
            return bins, func(z=z, MUV=bins)

        ##
        # Otherwise, standard SFE parameterized approach.
        ##

        # These are absolute AB magnitudes in `x_phi`
        x_phi, phi = self._get_phi_of_M(z, x=x, units=units, window=window)

        ok = phi.mask == False

        if ok.sum() == 0:
            return bins, -np.inf * np.ones_like(bins)

        # Setup interpolant. x_phi is in descending, remember!
        interp = interp1d(x_phi[ok][-1::-1], np.log10(phi[ok][-1::-1]),
            kind=self.pf['pop_interp_lf'],
            bounds_error=False, fill_value=np.log10(tiny_phi))

        if not absolute:
            bins_abs = self.get_mags_abs(z, bins)
        else:
            bins_abs = bins

        phi_of_x = 10**interp(bins_abs)

        return bins, phi_of_x

    def _get_uvlf_lum(self, LUV, z=None, x=1600., units='Angstroms', window=1):
        x_phi, phi = self._get_phi_of_L(z, x=x, units=units, window=window)

        ok = phi.mask == False

        if ok.sum() == 0:
            return -np.inf

        # Setup interpolant
        interp = interp1d(np.log10(x_phi[ok]), np.log10(phi[ok]),
            kind=self.pf['pop_interp_lf'],
            bounds_error=False, fill_value=np.log10(tiny_phi))

        phi_of_x = 10**interp(np.log10(LUV))

        return phi_of_x

    def get_uvlf(self, z, bins, use_mags=True, wave=1600., window=1.,
        absolute=True):
        return self.get_lf(z, bins, use_mags=use_mags, wave=wave,
            window=window, absolute=absolute)

    def get_lf(self, z, bins, use_mags=True, x=1600., units='Angstrom', window=1.,
        absolute=True, raw=False, nebular_only=False, band=None):
        """
        Reconstructed luminosity function.

        ..note:: This is number density per [abcissa].

        Parameters
        ----------
        z : int, float
            Redshift. Will interpolate between values in halos.tab_z if necessary.
        mags : bool
            If True, x-values will be in absolute (AB) magnitudes

        Returns
        -------
        Number density in # cMpc^-3 mag^-1.

        """

        if self.pf['pop_halos'] is not None:

            assert self.pf['pop_volume'] is not None

            if use_mags:
                x = self.get_mags(z=z, x=x, units=units, window=window,
                    absolute=absolute, raw=raw, nebular_only=nebular_only)
            else:
                x = self.get_lum(z=z, x=x, units=units, window=window,
                    raw=raw, nebular_only=nebular_only)

            phi, b_e = np.histogram(x, bins=bin_c2e(bins))
            return bins, phi / self.pf['pop_volume']

        if use_mags:
            _x_, phi_of_x = self._get_uvlf_mags(z, bins, x=x, units=units,
                window=window, absolute=absolute)
        else:
            # By default, we compute dn/dL
            bins, phi_of_x = self._get_phi_of_L(z, x=x, units=units,
                window=window, raw=raw, nebular_only=nebular_only, band=band)
            #raise NotImplemented('needs fixing')
            #phi_of_x = self._get_uvlf_lum(bins, z, wave=wave, window=window)

        return bins, phi_of_x

    def get_uvlf(self, z, bins):
        """
        Wrapper around `get_lf` to return what people usually mean by
        the UVLF, i.e., rest-UV = 1600 Angstrom, absolute AB magnitudes.
        """
        return self.get_lf(z, bins, use_mags=True, x=1600, units='Angstroms',
            absolute=True)

    def get_bias(self, z, limit, wave=1600., cut_in_flux=False,
        cut_in_mass=False, absolute=False):
        """
        Compute linear bias of galaxies brighter than (or more massive than)
        some cut-off.
        """
        iz = np.argmin(np.abs(z - self.halos.tab_z))

        tab_M = self.halos.tab_M
        tab_b = self.halos.tab_bias[iz,:]
        tab_n = self.halos.tab_dndm[iz,:]

        if cut_in_flux:
            raise NotImplemented('help')
        elif cut_in_mass:
            ok = tab_M >= limit
        else:
            ok = np.logical_and(mags <= limit, np.isfinite(mags))

        integ_top = tab_b[ok==1] * tab_n[ok==1]
        integ_bot = tab_n[ok==1]

        b = np.trapz(integ_top * tab_M[ok==1], x=np.log(tab_M[ok==1])) \
          / np.trapz(integ_bot * tab_M[ok==1], x=np.log(tab_M[ok==1]))

        return b

    def _cache_L(self, z, wave, band, window, units_out, raw, nebular_only, age):
        if not hasattr(self, '_cache_L_'):
            self._cache_L_ = {}

        if (z, wave, band, window, units_out, raw, nebular_only, age) in self._cache_L_:
            return self._cache_L_[(z, wave, band, window, units_out,
                raw, nebular_only, age)]

        return None

    def get_lum_per_sfr(self, z, Mh=None, x=1600., window=1., raw=False,
        nebular_only=False, band=None, units='Angstrom', age=None):
        """
        Return luminosity per unit SFR.
        """

        if not self.is_metallicity_constant:
            Z = self.get_metallicity(z, Mh=self.halos.tab_M)

            f_L_sfr = self._get_lum_all_Z(x=x, band=band,
                units=units,
                window=window, raw=raw, nebular_only=nebular_only, age=age)

            L_sfr = 10**f_L_sfr(np.log10(Z))

        elif self.pf['pop_lum_per_sfr'] is None:
            L_sfr = self.src.get_lum_per_sfr(x=x, window=window,
                band=band, units=units, raw=raw,
                nebular_only=nebular_only, age=age)
        else:
            assert self.pf['pop_calib_lum'] is None, \
                "# Be careful: if setting `pop_lum_per_sfr`, should leave `pop_calib_lum`=None."
            L_sfr = self.pf['pop_lum_per_sfr']

        return L_sfr

    def get_spec(self, z, waves, M=None, stellar_mass=False, per_Hz=False,
        band=None, window=1, units_out='erg/s/Hz', load=True, raw=False,
        nebular_only=False):
        """
        Compute the SED for all galaxies at given redshift.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        waves : np.ndarray
            Array of rest-wavelengths at which to generate SED [Angstrom].
        M : np.ndarray, optional
            If not supplied, will generate SED in pre-determined set of
            halo mass bins corresponding to `self.halos.tab_M`. Otherwise,
            will interpolate onto user-supplied range.
        stellar_mass : bool
            If True, assumes the user-supplied `M` values (above) are stellar
            masses, not halo masses, in which case the SMHM is first computed
            and interpolation is done in M_stell rather than M_h.


        Just a wrapper around `get_lum` with option of interpolating to new
        mass.
        """

        if M is None:
            assert not stellar_mass
            M = self.halos.tab_M

        if self.pf['pop_sfr_model'] in ['smhm-func', 'sfr-func']:

            # In this case, assume `M` supplied is stellar mass.
            # Convert to a halo mass via SMHM.
            if stellar_mass:
                # Mstell = Mhalo * SMHM
                smhm = self.get_smhm(z=z, Mh=self.halos.tab_M)
                mste = self.halos.tab_M * smhm
                Mh = np.interp(M, mste, self.halos.tab_M)
            else:
                Mh = M * 1.

            lum = np.zeros((M.size, waves.size))
            for i, wave in enumerate(waves):
                l = self.get_lum(z, x=wave, units='Angstroms',
                    band=band, window=window,
                    units_out=units_out, load=False, raw=raw,
                    nebular_only=nebular_only)
                lum[:,i] = 10**np.interp(np.log10(Mh), np.log10(self.halos.tab_M),
                    np.log10(l))

            if per_Hz:
                pass
            else:
                dwdn = waves**2 / (c * 1e8)
                lum = lum / dwdn[None,:]

            return lum

        else:
            raise NotImplemented('help')

    def get_spec_obs(self, z, waves=None,  M=None, per_Hz=False):
        if waves is None:
            waves = self.src.tab_waves_c
            dwdn = self.src.tab_dwdn
        else:
            dwdn = waves**2 / (c * 1e8)

        spec = self.get_spec(z, waves=waves, M=M, per_Hz=per_Hz)
        dL = self.cosm.get_luminosity_distance(z)

        # Flux at Earth in erg/s/cm^2/Hz
        f = spec / (4. * np.pi * dL**2)

        # Correct for redshifting and change in units.
        if per_Hz:
            f *= (1. + z)
        else:
            f /= dwdn
            f /= (1. + z)

        owaves = waves * (1. + z) / 1e4

        return owaves, f

    def get_lum(self, z, x=1600, band=None, window=1, units='Angstrom',
        units_out='erg/s/A', load=True, raw=False, nebular_only=False, age=None):
        """
        Return the luminosity of all halos at given redshift `z`.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        x : int, float
            Wavelength or photon energy or photon frequency, set by `units`.
        band : 2-element tuple
            Defines edge of band, if interest in band-averaged luminosity
            rather than monochromatic luminosity. Abides by `units` keyword
            as well. Note: will override `x` if both are provided!

        Returns
        -------
        Array of luminosities corresponding to halos in model.

        """

        #if False:
        kwtup = (z, x, band, window, units_out, raw, nebular_only, age)
        cached_result = self._cache_L(*kwtup)

        if cached_result is not None:
            return cached_result

        ##
        # Can load halos from catalog instead of HMF tables [optional]
        if self.pf['pop_halos'] is not None:
            Mh, _x, _y, _z = self.pf['pop_halos'](z=z).T

            if not np.all(np.diff(Mh) > 0):
                Mh = Mh[-1::-1]
                flipM = True
        else:
            Mh = self.halos.tab_M

        ##
        # Have options for stars or BHs
        if self.pf['pop_star_formation']:

            if self.pf['pop_ssp'] and self.pf['pop_age'] is not None:

                age = self.get_age(z=z, Mh=self.halos.tab_M)

                if type(age) in numeric_types:
                    age = age * np.ones_like(self.halos.tab_M)

                # Assert that age < Hubble time at this redshift
                t_H = self.cosm.HubbleTime(z) / s_per_myr
                age[age > t_H] = t_H
                if np.any(age > t_H):
                    print(f"WARNING: some ages exceed Hubble time at z={z}.")

            if not self.is_metallicity_constant:
                if age is not None:
                    raise NotImplemented("This is probably broken.")

                Z = self.get_metallicity(z, Mh=self.halos.tab_M)

                f_L_sfr = self._get_lum_all_Z(x=x, band=band,
                    units=units, window=window, raw=raw,
                    nebular_only=nebular_only, age=age)

                L_sfr = 10**f_L_sfr(np.log10(Z))

                #L_sfr = self.get_lum_per_sfr(wave=wave, #Z=Z,
                #    window=window, band=band, band_units=band_units, raw=raw,
                #    nebular_only=nebular_only)

            elif self.pf['pop_lum_per_sfr'] is None:
                if age is not None:
                    L_sfr = np.array([self.src.get_lum_per_sfr(x=x,
                        window=window, band=band, units=units, raw=raw,
                        nebular_only=nebular_only, age=_age_,
                        units_out=units_out) for _age_ in age])
                else:
                    L_sfr = self.src.get_lum_per_sfr(x=x, window=window,
                        band=band, units=units, units_out=units_out,
                        raw=raw, nebular_only=nebular_only)
            else:
                assert self.pf['pop_calib_lum'] is None, \
                    "# Be careful: if setting `pop_lum_per_sfr`, should leave `pop_calib_lum`=None."
                L_sfr = self.pf['pop_lum_per_sfr']

            if self.is_quiescent:

                assert self.pf['pop_ssp']

                # Mstell = Mhalo * SMHM
                smhm = self.get_smhm(z=z, Mh=self.halos.tab_M)
                mste = self.halos.tab_M * smhm

                if self.is_central_pop:

                    # Not for SSPs, L per SFR is really L per Mstell.
                    Lh = mste * L_sfr

                    if self.pf['pop_ihl'] is not None:
                        ihl = self.ihl(z=z, Mh=self.halos.tab_M)
                        Lh *= ihl
                else:

                    iz = np.argmin(np.abs(z - self.halos.tab_z))
                    fsurv = self.tab_fsurv[iz,:]

                    # In this case, need to integrate over subhalo MF.
                    Ls = mste * L_sfr
                    Lh = np.zeros_like(self.halos.tab_M)
                    for i, Mc in enumerate(self.halos.tab_M):
                        dndlnm = self.halos.tab_dndlnm_sub[i,:] * fsurv
                        Lh[i] = np.trapz(Ls * dndlnm, x=np.log(self.halos.tab_M))


            else:

                # This uses __getattr__ in case we're allowing Z to be
                # updated from SAM.
                sfr = self.get_sfr(z)

                # Just the product of SFR and L
                if self.is_central_pop:
                    Lh = sfr * L_sfr
                else:
                    iz = np.argmin(np.abs(z - self.halos.tab_z))
                    fsurv = self.tab_fsurv[iz,:]

                    # In this case, need to integrate over subhalo MF.
                    Ls = sfr * L_sfr
                    Lh = np.zeros_like(self.halos.tab_M)
                    for i, Mc in enumerate(self.halos.tab_M):
                        dndlnm = self.halos.tab_dndlnm_sub[i,:] * fsurv
                        Lh[i] = np.trapz(Ls * dndlnm, x=np.log(self.halos.tab_M))

            ok = np.logical_and(self.halos.tab_M >= self.get_Mmin(z),
                self.halos.tab_M < self.get_Mmax(z))
            Lh[~ok] = 0

            ##
            # Apply fesc
            fesc = self.get_fesc(z, Mh=self.halos.tab_M, x=x, band=band,
                units=units)
            Lh *= fesc

            ##
            # Apply dust reddening
            if not self.is_dusty:
                T = 1
            else:
                # Depending on approach, either need to first get Av, dust
                # surface density, or MUV-Beta/IRXB
                wave = self.src.get_ang_from_x(x, units=units)

                smhm = self.get_smhm(z=z, Mh=self.halos.tab_M)
                Ms = self.halos.tab_M * smhm
                if self.pf['pop_dust_template'] is not None:
                    Av = self.get_Av(z=z, Ms=Ms)
                    Sd = None
                elif self.pf['pop_dust_yield'] is not None:
                    Av = None
                    Sd = self.get_dust_surface_density(z, Mh=self.halos.tab_M)
                else:
                    raise NotImplemented('help')

                T = self.dust.get_transmission(wave, Av=Av, Sd=Sd)
                Lh *= T.squeeze()

            ##
            # Apply IGM filtering here eventually. Actually, that could be
            # dangerous -- perhaps should only ever be done in post.

            if not hasattr(self, '_cache_L'):
                self._cache_L = {}

            self._cache_L_[kwtup] = Lh

            return Lh

        elif self.pf['pop_bh_formation']:
            # In this case, luminosity just proportional to BH mass.
            zarr, data = self.get_histories()

            iz = np.argmin(np.abs(zarr - z))

            # Interpolate Mbh onto halo mass grid so we can use abundances.
            Mbh = np.exp(np.interp(np.log(self.halos.tab_M),
                np.log(data['Mh'][:,iz]),
                np.log(data['Mbh'][:,iz])))

            # Bolometric luminosity: Eddington
            ledd = 4 * np.pi * G * m_p * c / sigma_T
            Lbol = ledd * Mbh * g_per_msun
            Lbol[np.isnan(Lbol)]= 0.0

            # Need to do bolometric correction.
            E = h_p * c / (wave * 1e-8) / erg_per_ev
            I_E = self.src.get_spectrum(E)

            Lh = Lbol * I_E * ev_per_hz

            #self._cache_L_[(z, wave)] = Lh

            return Lh

            # Don't need to do trajectories unless we're letting
            # BHs grow via accretion, i.e., scaling laws can just get
            # painted on.

        else:
            raise NotImplemented('help')

    def get_Av(self, z, Ms):
        """
        Get visual extinction.
        """

        #assert self.pf['pop_dustext_template'] is not None

        func = self._get_function('pop_Av')

        # Should set this up to convert from dust opacity too.

        return func(z=z, Ms=Ms)

        #Mh = self.halos.tab_M
        #smhm = self.get_smhm(z=z, Mh=Mh)
        #Ms = Mh * smhm

        #return self._func_Av(z=z, Ms=Ms)

    def get_age(self, z, Mh):
        func = self._get_function('pop_age')
        return func(z=z, Mh=Mh)

    def get_Nion(self, z, Mh):
        func = self._get_function('pop_Nion')
        return func(z=z, Mh=Mh)

    def get_dust_yield(self, z, Mh):
        """
        Return fraction of metals locked up in dust grains.
        """

        func = self._get_function('pop_dust_yield')
        result = func(z=z, Mh=Mh)
        return result

    def get_dust_scale(self, z, Mh):
        """
        Return dust scale length in kiloparsecs.
        """

        func = self._get_function('pop_dust_scale')
        result = func(z=z, Mh=Mh)
        return result

    def get_dust_scatter(self, z, Mh):
        """
        Return dust scale length in kiloparsecs.
        """

        func = self._get_function('pop_dust_scatter')
        result = func(z=z, Mh=Mh)
        return result

    def get_dust_fcov(self, z, Mh):
        """
        Return dust scale length in kiloparsecs.
        """

        func = self._get_function('pop_dust_fcov')
        result = func(z=z, Mh=Mh)
        return result

    def get_dust_surface_density(self, z, Mh):
        fb = self.cosm.fbaryon
        fmr = self.pf['pop_mass_yield']
        fZy = fmr * self.pf['pop_metal_yield']
        fd = self.get_dust_yield(z=z, Mh=Mh)
        if self.is_user_smhm:

            smhm = self.get_smhm(z=z, Mh=Mh)
            smhm[Mh < self.get_Mmin(z)] = 0
            smhm[Mh > self.get_Mmax(z)] = 0
            Ms = smhm * Mh
        else:
            raise NotImplemented('help')
        Md = fd * fZy * Ms
        Rd = self.get_dust_scale(z=z, Mh=Mh)
        # Assumes spherical symmetry, uniform dust density
        Sd = 3. * Md * g_per_msun \
            / 4. / np.pi / (Rd * cm_per_kpc)**2

        return Sd

    def get_mags(self, z, absolute=True, x=1600, band=None,
        units='Angstrom', window=1, cam=None, filters=None,
        filter_set=None, presets=None,
        load=True, raw=False, nebular_only=False, apply_dustcorr=False):
        """
        Return magnitudes corresponding to halos in model at redshift `z`.

        .. note :: Assumes AB magnitudes, either absolute or apparent
            depending on value of `absolute` keyword argument.

        """

        if presets is not None:
            filter_set = None
            cam, filters = self._get_presets(z, presets)

        if type(filters) == dict:
            filters = filters[round(z)]

        if type(filters) == str:
            filters = (filters, )

        L = self.get_lum(z, x=x, band=band, units=units,
            window=window, units_out='erg/s/Hz', load=load, raw=raw,
            nebular_only=nebular_only)

        mags = self.magsys.L_to_MAB(L)

        if absolute:
            return mags
        else:
            if apply_dustcorr:
                raise NotImplemented('help!')

            return self.get_mags_app(z, mags)

    def get_Mmax_from_maglim(self, z, x, mlim, mtol=0.05):
        """
        Map some limiting (apparent AB) magnitude at given wavelength onto a
        corresponding halo mass.

        Parameters
        ----------
        z : int, float
            Redshift of object.
        x : int, float
            Observed wavelength [microns].
        mlim : int, float
            Apparent AB magnitude of interest.

        Returns
        -------
        Halo mass in Msun.
        """
        if type(wave) in numeric_types:
            lam_r = wave * 1e4 / (1. + z)
            mags = self.get_mags(z, absolute=False, x=lam_r, window=51,
                units='Angstrom')
        else:
            raise NotImplemented('help')

        ok = np.isfinite(mags)
        if ok.sum() == 0:
            return mags, self.halos.tab_M.max()

        Mh_lim = 10**np.interp(mlim, mags[ok==1][-1::-1],
            np.log10(self.halos.tab_M[ok==1])[-1::-1],
            left=np.log10(self.halos.tab_M.max()),
            right=np.log10(self.halos.tab_M.min()))

        return mags, Mh_lim

    def get_mags_tab(self, wave):

        mags = np.inf * np.ones_like(self.halos.tab_dndm)
        for i, z in enumerate(self.halos.tab_z):
            # last argument doesn't matter here.
            mags[i,:], Mhlim = self.get_Mmax_from_maglim(z, wave, 15)

        return mags

    @property
    def tab_source_mask(self):
        """
        Halo mass corresponding to some limiting magnitude set in pop_mask.
        """
        if not hasattr(self, '_tab_source_mask'):
            self._tab_source_mask = self.halos.tab_M.max() \
                * np.ones_like(self.halos.tab_z)

            if self.pf['pop_mask'] is None:
                return self._tab_source_mask

            tmp = self._tab_source_mask.copy()

            for i, z in enumerate(self.halos.tab_z):

                if self.pf['pop_mask_interp'] is not None:
                    if i % self.pf['pop_mask_interp'] != 0:
                        continue

                # Loop over masking thresholds
                Mh_lim = self.halos.tab_M.max()
                for j, mask in enumerate(self.pf['pop_mask']):
                    if len(mask) == 2:
                        mwave, mlim = mask
                        _mags, Mh_lim_j = self.get_Mmax_from_maglim(z, mwave, mlim)
                        Mh_lim = min(Mh_lim, Mh_lim_j)
                    else:
                        mwave, mlim, mags = mask
                        # inf means 0 = flux, set to some impossibly faint
                        # magnitude to avoid issues
                        if not np.any(np.isfinite(mags[i,:])):
                            continue

                        ok = np.isfinite(mags[i])
                        if ok.sum() == 0:
                            continue

                        cand = 10**np.interp(mlim, mags[i,ok==1][-1::-1],
                            np.log10(self.halos.tab_M[ok==1])[-1::-1],
                            left=np.log10(self.halos.tab_M.max()),
                            right=np.log10(self.halos.tab_M.min()))

                        Mh_lim = min(Mh_lim, cand)

                tmp[i] = Mh_lim

            if self.pf['pop_mask_interp'] is not None:
                skip = self.pf['pop_mask_interp']
                self._tab_source_mask = 10**np.interp(self.halos.tab_z,
                    self.halos.tab_z[::skip], np.log10(tmp[::skip]))
            else:
                self._tab_source_mask = tmp

        return self._tab_source_mask

    def _get_phi_of_L(self, z, x=1600., window=1, raw=False,
        nebular_only=False, band=None, units=None):
        """
        Compute the luminosity function at redshift z.

        Returns
        -------
        Number of galaxies per unit luminosity per unit volume.

        """

        if not hasattr(self, '_phi_of_L'):
            self._phi_of_L = {}
        else:
            if (z, x, window) in self._phi_of_L:
                return self._phi_of_L[(z, x, window)]

        Lh = self.get_lum(z, x=x, window=window,
            raw=raw, nebular_only=nebular_only, band=band, units=units)

        if self.pf['pop_halos'] is None:
            Mh = self.halos.tab_M
        else:
            Mh, _x, _y, _z = self.pf['pop_halos'](z=z).T

        #fobsc = (1. - self.get_fobsc(z=z, Mh=Mh))
        # Means obscuration refers to fractional dimming of individual
        # objects
        #if self.pf['pop_fobsc_by'] == 'lum':
        #    Lh *= fobsc

        logL_Lh = np.log(Lh)

        ##
        # Continue with standard approach.
        iz = np.argmin(np.abs(z - self.halos.tab_z))

        if abs(z - self.halos.tab_z[iz]) < ztol:
            dndm = self.halos.tab_dndm[iz,:-1] * self.tab_focc[iz,:-1]
        else:
            dndm_func = interp1d(self.halos.tab_z, self.halos.tab_dndm[:,:-1],
                axis=0, kind=self.pf['pop_interp_lf'])

            dndm = dndm_func(z) * self.get_focc(z=z, Mh=self.halos.tab_M[0:-1])

        # In this case, obscuration means fraction of objects you don't see
        # in the UV.
        if self.pf['pop_fobsc_by'] == 'num':
            dndm *= fobsc[0:-1]

        dMh_dLh = np.diff(self.halos.tab_M) / np.diff(Lh)

        dMh_dlogLh = dMh_dLh * Lh[0:-1]

        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.tab_z, self._tab_Mmin)
        Mmax = self.pf['pop_lf_Mmax']

        i_min = np.argmin(np.abs(Mmin - self.halos.tab_M))
        i_max = np.argmin(np.abs(Mmax - self.halos.tab_M))

        if self.pf['pop_Lh_scatter'] > 0:
            sigma = self.pf['pop_Lh_scatter']
            norm = np.sqrt(2. * np.pi) / sigma / np.log(10.)

            gauss = lambda x, mu: np.exp(-(x - mu)**2 / 2. / sigma**2) / norm

            phi_of_L = np.zeros_like(Lh[0:-1])
            for k, logL in enumerate(logL_Lh[0:-1]):

                # Actually a range of halo masses that can produce galaxy
                # of luminosity Lh
                pdf = gauss(logL_Lh[0:-1], logL_Lh[k])

                integ = dndm[i_min:i_max] * pdf[i_min:i_max] * dMh_dlogLh[i_min:i_max]

                phi_of_L[k] = np.trapz(integ, x=logL_Lh[i_min:i_max])

            # This needs extra term now?
            phi_of_L /= Lh[0:-1]

        else:
            phi_of_L = dndm * dMh_dLh

        above_Mmin = self.halos.tab_M >= Mmin
        below_Mmax = self.halos.tab_M <= Mmax
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)

        lum = np.ma.array(Lh[:-1], mask=mask)
        phi = np.ma.array(phi_of_L, mask=mask, fill_value=tiny_phi)

        self._phi_of_L[(z, x, window)] = lum, phi

        return self._phi_of_L[(z, x, window)]

    def _get_phi_of_M(self, z, x=1600., units='Angstroms', window=1):
        if not hasattr(self, '_phi_of_M'):
            self._phi_of_M = {}
        else:
            if (z, x, window) in self._phi_of_M:
                return self._phi_of_M[(z, x, window)]

        Lh, phi_of_L = self._get_phi_of_L(z, x=x, units=units, window=window)

        _MAB = self.magsys.L_to_MAB(Lh)

        wave = self.src.get_ang_from_x(x, units=units)
        if (self.pf['dustcorr_method'] is not None) and wave <= 1600:
            MAB = self.dust.Mobs(z, _MAB)
        else:
            MAB = _MAB

        if self.pf['pop_halos'] is None:
            phi_of_M = phi_of_L[0:-1] * np.abs(np.diff(Lh) / np.diff(MAB))
            phi_of_M[phi_of_M==0] = tiny_phi
        else:
            raise NotImplemented('help')

        self._phi_of_M[(z, x, window)] = MAB[0:-1], phi_of_M

        return self._phi_of_M[(z, x, window)]

    def get_mag_lim(self, z, absolute=True, x=1600, band=None, units='Ang',
        window=1, load=True, raw=False, nebular_only=False, apply_dustcorr=False):
        """
        Compute the magnitude corresponding to the minimum mass threshold.
        """

        mags = self.get_mags(z, absolute=absolute, x=x, units=units, band=band,
            window=window, load=load, raw=raw,
            nebular_only=nebular_only, apply_dustcorr=apply_dustcorr)

        Mmin = self.get_Mmin(z)

        ok = np.isfinite(mags)
        return np.interp(Mmin, self.halos.tab_M[ok==1], mags[ok==1])

    def get_Mmax(self, z):
        # Doesn't have a setter because of how we do things in Composite.
        # Long story.
        return np.interp(z, self.halos.tab_z, self._tab_Mmax)

    @property
    def tab_Matom(self):
        if not hasattr(self, '_tab_Matom'):
            Mvir = lambda z: self.halos.VirialMass(z, 1e4, mu=self.pf['mu'])
            self._tab_Matom = np.array([Mvir(z) for z in self.halos.tab_z])
        return self._tab_Matom

    def get_Mmin(self, z):
        """
        Return minimum halo mass for given redshift `z`.

        .. note :: First makes interpolant over tabulated Mmin(z) grid, then
            calls that function on subsequent calls.

        """
        if not hasattr(self, '_func_Mmin'):
            self._func_Mmin = lambda z: \
                np.interp(z, self.halos.tab_z, self._tab_Mmin)

        return self._func_Mmin(z)

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            self._Mmin = lambda z: \
                np.interp(z, self.halos.tab_z, self._tab_Mmin)

        return self._Mmin

    @Mmin.setter
    def Mmin(self, value):
        if ismethod(value):
            self._Mmin = value
        else:
            self._tab_Mmin = value
            self._Mmin = lambda z: np.interp(z, self.halos.tab_z, self._tab_Mmin)

    def Mmax(self, z):
        # Doesn't have a setter because of how we do things in Composite.
        # Long story.
        return np.interp(z, self.halos.tab_z, self._tab_Mmax)

    @property
    def _tab_logMmin(self):
        if not hasattr(self, '_tab_logMmin_'):
            self._tab_logMmin_ = np.log(self._tab_Mmin)
        return self._tab_logMmin_

    @property
    def _tab_logMmax(self):
        if not hasattr(self, '_tab_logMmax_'):
            self._tab_logMmax_ = np.log(self._tab_Mmax)
        return self._tab_logMmax_

    @property
    def _loaded_guesses(self):
        if not hasattr(self, '_loaded_guesses_'):
            self._loaded_guesses_ = False
        return self._loaded_guesses_

    @_loaded_guesses.setter
    def _loaded_guesses(self, value):
        self._loaded_guesses_ = value

    @property
    def _tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin_'):

            # First, compute threshold mass vs. redshift
            if self.pf['feedback_LW_guesses'] is not None and (not self._loaded_guesses):
                guess = self._guess_Mmin()
                if guess is not None:
                    self._tab_Mmin = guess
                    return self._tab_Mmin_

            if self.pf['pop_Mmin'] is not None:
                if ismethod(self.pf['pop_Mmin']) or \
                   (type(self.pf['pop_Mmin']) == FunctionType):
                    self._tab_Mmin_ = \
                        np.array(map(self.pf['pop_Mmin'], self.halos.tab_z))
                elif type(self.pf['pop_Mmin']) is np.ndarray:
                    self._tab_Mmin_ = self.pf['pop_Mmin']
                    assert self._tab_Mmin.size == self.halos.tab_z.size
                elif type(self.pf['pop_Mmin']) is str:
                    if self.pf['pop_Mmin'] == 'jeans':
                        self._tab_Mmin_ = \
                            np.array(map(self.cosm.JeansMass, self.halos.tab_z))
                    elif self.pf['pop_Mmin'] == 'filtering':
                        self._tab_Mmin_ = \
                            np.array(map(self.halos.FilteringMass, self.halos.tab_z))
                    else:
                        raise NotImplemented('help')
                else:
                    self._tab_Mmin_ = self.pf['pop_Mmin'] \
                        * np.ones_like(self.halos.tab_z)

            else:
                self._tab_Mmin_ = self.halos.VirialMass(
                    self.halos.tab_z, self.pf['pop_Tmin'], mu=self.pf['mu'])

            self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, 'min')

        return self._tab_Mmin_

    @_tab_Mmin.setter
    def _tab_Mmin(self, value):
        if ismethod(value):
            self._func_Mmin = value
            self._tab_Mmin_ = np.array([value(z) for z in self.halos.tab_z],
                dtype=float)
        elif type(value) in [int, float, np.float64]:
            self._tab_Mmin_ = value * np.ones_like(self.halos.tab_z)
        else:
            self._tab_Mmin_ = value

        self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, s='min')

    @property
    def _spline_ngtm(self):
        if not hasattr(self, '_spline_ngtm_'):
            # Need to setup spline for n(>M)
            log10_ngtm = np.log10(self.halos.tab_ngtm)
            not_ok = np.isinf(log10_ngtm)
            ok = np.logical_not(not_ok)

            log10_ngtm[ok==0] = -40.

            _spl = RectBivariateSpline(self.halos.tab_z,
               np.log10(self.halos.tab_M), log10_ngtm)
            self._spline_ngtm_  = \
                lambda z, log10M: 10**_spl(z, log10M).squeeze()

        return self._spline_ngtm_

    @property
    def _tab_n_Mmin(self):
        """
        Number of objects in each Mmin bin. Only use this for setting
        up Ensemble objects?
        """
        if not hasattr(self, '_tab_n_Mmin_'):

            # Interpolate halo abundances onto Mmin axis.
            ngtm_Mmin = np.array([self._spline_ngtm(self.halos.tab_z[i],
                np.log10(self._tab_Mmin)[i]) \
                    for i in range(self.halos.tab_z.size)])

            # Number of halos in this Mmin bin is just the difference
            # in N(M>Mmin) between two redshift steps.
            # Remember, though, that ngtm_Mmin is in *descending* order
            # since it rises with redshift, hence the minus sign.
            n_new = np.concatenate((-np.diff(ngtm_Mmin), [0.0]))

            self._tab_n_Mmin_ = n_new

        return self._tab_n_Mmin_

    @property
    def _tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor_'):
            self._tab_Mmin_floor_ = self.halos.Mmin_floor(self.halos.tab_z)
        return self._tab_Mmin_floor_

    def _apply_lim(self, arr, s='min', zarr=None):
        """
        Adjust Mmin or Mmax so that Mmax > Mmin and/or obeys user-defined
        floor and ceiling.
        """
        out = None

        if zarr is None:
            zarr = self.halos.tab_z

        # Might need these if Mmin is being set dynamically
        if self.pf['pop_M{!s}_ceil'.format(s)] is not None:
            out = np.minimum(arr, self.pf['pop_M{!s}_ceil'.format(s)])
        if self.pf['pop_M{!s}_floor'.format(s)] is not None:
            out = np.maximum(arr, self.pf['pop_M{!s}_floor'.format(s)])
        if self.pf['pop_T{!s}_ceil'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(z,
                self.pf['pop_T{!s}_ceil'.format(s)], mu=self.pf['mu'])
            _MofT = np.array(list(map(_f, zarr)))
            out = np.minimum(arr, _MofT)
        if self.pf['pop_T{!s}_floor'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(z,
                self.pf['pop_T{!s}_floor'.format(s)], mu=self.pf['mu'])
            _MofT = np.array(list(map(_f, zarr)))
            out = np.maximum(arr, _MofT)

        if out is None:
            out = arr.copy()

        # Impose a physically-motivated floor to Mmin as a last resort,
        # by default this will be the Tegmark+ limit.
        if s == 'min':
            out = np.maximum(out, self._tab_Mmin_floor)

        return out

    @property
    def _done_setting_Mmax(self):
        if not hasattr(self, '_done_setting_Mmax_'):
            self._done_setting_Mmax_ = False
        return self._done_setting_Mmax_

    @property
    def _tab_Mmax(self):
        if not hasattr(self, '_tab_Mmax_'):

            # First, compute threshold mass vs. redshift
            t_limit = self.pf['pop_time_limit']
            m_limit = self.pf['pop_mass_limit']
            a_limit = self.pf['pop_abun_limit']
            e_limit = self.pf['pop_bind_limit']
            T_limit = self.pf['pop_temp_limit']

            if t_limit == 0:
                t_limit = None
            if e_limit == 0:
                e_limit = None

            if (t_limit is not None) or (m_limit is not None) or \
               (e_limit is not None) or (T_limit is not None) or \
               (a_limit is not None):

                M0x = self.pf['pop_initial_Mh']
                if (M0x == 0) or (M0x == 1):
                    zform, zfin, Mfin, raw = self.MassAfter()
                    new_data = self._sort_sam(self.zform,
                        zform, raw, sort_by='form')

                    self.tmp_data = new_data

                else:
                    zform, zfin, Mfin, raw = self.MassAfter(M0=M0x)
                    new_data = self._sort_sam(self.zform,
                        zform, raw, sort_by='form')

                # This is the redshift at which the first star-forming halo,
                # formed at (zi, M0), transitions to PopII.
                zmax = max(zfin)

                # This is the mass trajectory of a halo that forms at
                # initial_redshift with initial mass pop_initial_Mh
                # (in units of Mmin, defaults to 1).
                Moft_zi = lambda z: np.interp(z, zform, new_data['Mh'])

                # For each redshift, determine Mmax.
                Mmax = np.zeros_like(self.halos.tab_z)
                for i, z in enumerate(self.halos.tab_z):

                    # If we've specified a maximum initial mass halo, and
                    # we're at a redshift before that halo hits its limit.
                    # Or, we're using a time-limited model.
                    if ((M0x > 0) and (z > zmax)):
                        Mmax[i] = Moft_zi(z)
                    else:
                        Mmax[i] = 10**np.interp(z, zfin, np.log10(Mfin))

                self._tab_Mmax_ = Mmax

            elif self.pf['pop_Mmax'] is not None:
                if type(self.pf['pop_Mmax']) is FunctionType:
                    self._tab_Mmax_ = np.array(list(map(self.pf['pop_Mmax'], self.halos.tab_z)))

                elif type(self.pf['pop_Mmax']) is tuple:
                    extra = self.pf['pop_Mmax'][0]
                    assert self.pf['pop_Mmax'][1] == 'Mmin'

                    if type(extra) is FunctionType:
                        self._tab_Mmax_ = np.array(list(map(extra, self.halos.tab_z))) \
                            * self._tab_Mmin
                    else:
                        self._tab_Mmax_ = extra * self._tab_Mmin
                else:
                    self._tab_Mmax_ = self.pf['pop_Mmax'] * np.ones_like(self.halos.tab_z)

            elif self.pf['pop_Tmax'] is not None:
                Mvir = lambda z: self.halos.VirialMass(z, self.pf['pop_Tmax'],
                    mu=self.pf['mu'])
                self._tab_Mmax_ = np.array(list(map(Mvir, self.halos.tab_z)))

            else:
                # A suitably large number for (I think) any purpose
                self._tab_Mmax_ = 1e18 * np.ones_like(self.halos.tab_z)

            # Override switch: source masking
            if self.pf['pop_mask'] is not None:
                # Don't understand this...if tab_source_mask gets called
                # before this method, things don't work out. 03.15.2023.
                if hasattr(self, '_tab_source_mask'):
                    del self._tab_source_mask
                self._tab_Mmax_ = self.tab_source_mask

            self._tab_Mmax_ = self._apply_lim(self._tab_Mmax_, s='max')
            self._tab_Mmax_ = np.maximum(self._tab_Mmax_, self._tab_Mmin)

        return self._tab_Mmax_

    @_tab_Mmax.setter
    def _tab_Mmax(self, value):
        if type(value) in [int, float, np.float64]:
            self._tab_Mmax_ = value * np.ones_like(self.halos.tab_z)
        else:
            self._tab_Mmax_ = value

    @property
    def _tab_sfr_mask(self):
        if not hasattr(self, '_tab_sfr_mask_'):
            # Mmin is like tab_z, make it like (z, M)
            # M is like tab_M, make it like (z, M)
            Mmin = np.array([self._tab_Mmin] * self.halos.tab_M.size).T
            Mmax = np.array([self._tab_Mmax] * self.halos.tab_M.size).T
            M = np.reshape(np.tile(self.halos.tab_M, self.halos.tab_z.size),
                    (self.halos.tab_z.size, self.halos.tab_M.size))

            mask = np.zeros((self.halos.tab_z.size, self.halos.tab_M.size),
                dtype=bool)
            mask[M < Mmin] = True
            mask[M > Mmax] = True
            mask[self.halos.tab_z > self.zform] = True
            mask[self.halos.tab_z < self.zdead] = True
            self._tab_sfr_mask_ = mask
        return self._tab_sfr_mask_

    def _ngtm_from_ham(self, z):
        if not hasattr(self, '_ngtm_from_ham_'):
            self._ngtm_from_ham_ = {}

        if z in self._ngtm_from_ham_:
            return self._ngtm_from_ham_[z]

        # Compute n(>m) from discrete set of halos.
        hist = self.pf['pop_histories']
        iz = np.argmin(np.abs(hist['z'] - z))
        _Mh_ = hist['Mh'][:,iz]
        nh = hist['nh'][:,iz]

        # Make ngtm so it looks like it came from self.halos
        self._ngtm_from_ham_[z] = np.zeros_like(self.halos.tab_M)
        for iM, _M_ in enumerate(self.halos.tab_M):
            self._ngtm_from_ham_[z][iM] = np.sum(nh[_Mh_ > _M_])

        return self._ngtm_from_ham_[z]

    def get_AUV(self, z, MUV):
        """
        Return extinction in rest-UV at redshift `z` for absolute magnitude(s)
        `MUV`.

        .. note :: Just a wrapper around `self.dust.AUV`, which is using
            empirical dust corrections.

        """
        return self.dust.get_attenuation(z, MUV)

    def run_abundance_match(self, z, Mh, uvlf=None, x=1600., units='Angstroms'):
        """
        These are the star-formation efficiencies derived from abundance
        matching.
        """

        if self.pf['pop_ham_z'] is not None:
            z_ham = self.pf['pop_ham_z']

            if hasattr(self, '_ham_results'):

                _Mh, _fstar = self._ham_results

                if not _Mh.size == Mh.size:
                    fstar = np.exp(np.interp(np.log(Mh), np.log(_Mh),
                        np.log(_fstar)))
                else:
                    fstar = _fstar

                # Enforce minimum mass
                fstar[Mh < self.get_Mmin(z)] = 0.0

                return fstar
        else:
            z_ham = z

        mags_obs = np.arange(self.pf['pop_mag_min'],
            self.pf['pop_mag_max']+self.pf['pop_mag_bin'], self.pf['pop_mag_bin'])

        # Poke PQ
        if uvlf is None:
            assert self.pf['pop_sfr_model'] in ['uvlf', 'ham']
            def uvlf(z, mag):
                _x_, _phi_ = self.get_lf(z, mags_obs)

                return np.interp(mag, _x_, _phi_)

        else:
            assert type(uvlf) is FunctionType, "Must supply `uvlf` as function!"

        mags = []
        for mag in mags_obs:
            mags.append(mag-self.get_AUV(z_ham, mag))

        # Mass function
        if self.pf['pop_histories'] is not None:
            iz = np.argmin(np.abs(self.pf['pop_histories']['z'] - z_ham))
            ngtm = self._ngtm_from_ham(z_ham)
        else:
            iz = np.argmin(np.abs(z_ham - self.halos.tab_z))
            ngtm = self.halos.tab_ngtm[iz]

        # Use dust-corrected magnitudes here
        LUV_dc = np.array([self.magsys.MAB_to_L(mag) for mag in mags])

        #assert self.pf['pop_lum_per_sfr'] is not None
        if self.pf['pop_lum_per_sfr'] is not None:
            L_per_sfr = self.pf['pop_lum_per_sfr']
        else:
            L_per_sfr = self.src.get_lum_per_sfr(x=x, units=units,
                units_out='erg/s/Hz')

        # Loop over luminosities and perform abundance match
        mh_of_mag = []
        fstar_tab = []
        for j, _mag_ in enumerate(mags_obs):

            # Number of galaxies with MUV <= _mag_
            int_phiM = quad(lambda xx: uvlf(z=z_ham, mag=xx), -np.inf,
                _mag_)[0]

            # Number density of halos at masses > M
            ngtM_spl = interp1d(np.log10(self.halos.tab_M),
                np.log10(ngtm), kind='linear', bounds_error=False)

            def to_min(logMh):
                int_nMh = 10**(ngtM_spl(logMh))[0]

                return abs(int_phiM - int_nMh)

            _Mh_ = 10**fsolve(to_min, 10., factor=0.001,
                maxfev=10000)[0]

            if self.pf['pop_histories'] is not None:
                # Compute mean MAR for halos in this bin?
                _Mh_hist = np.log10(self.pf['pop_histories']['Mh'][:,iz])
                ok = np.logical_and(_Mh_hist >= np.log10(_Mh_)-0.05,
                                    _Mh_hist <  np.log10(_Mh_)+0.05)
                MAR = np.mean(self.pf['pop_histories']['MAR_acc'][:,iz])
            else:
                MAR = 10**np.interp(np.log10(_Mh_), np.log10(self.halos.tab_M),
                    np.log10(self.halos.tab_MAR[iz,:]))

            MAR *= self.cosm.fbar_over_fcdm

            print('hi', j, _mag_, LUV_dc[j], int_phiM, ngtm)

            _fstar_ = LUV_dc[j] / L_per_sfr / MAR

            mh_of_mag.append(_Mh_)
            fstar_tab.append(_fstar_)

        ##
        # Interpolate results onto provided Mh
        fstar = 10**np.interp(np.log10(Mh), np.log10(mh_of_mag[-1::-1]),
            np.log10(fstar_tab[-1::-1]))

        # In this case, cache
        if self.pf['pop_ham_z'] is not None:
            self._ham_results = Mh, fstar

        # Enforce minimum mass
        fstar[Mh < self.get_Mmin(z)] = 0.0

        return fstar

    def get_ssfr(self, z, Ms):
        """
        Compute specific star formation rate.
        """

        if self.is_user_smhm:
            func = self._get_function('pop_ssfr')
            return func(z=z, Ms=Ms)
        else:
            raise NotImplemented('help')

    @property
    def tab_ssfr(self):
        """
        Table of specific star formation rate as a function of redshift and
        *stellar* mass.
        """

        if not hasattr(self, '_tab_ssfr_'):
            assert self.pf['pop_sfr_model'] == 'smhm-func'

            Ms = self.tab_fstar * self.halos.tab_M

            #pars = get_pq_pars(self.pf['pop_ssfr'], self.pf)
            #_ssfr_inst = ParameterizedQuantity(**pars)
            #_ssfr = lambda **kwargs: _ssfr_inst.__call__(**kwargs)

            ssfr = np.zeros((self.halos.tab_z.size, self.halos.tab_M.size))
            for i, z in enumerate(self.halos.tab_z):
                fstar = self.get_fstar(z=z, Mh=self.halos.tab_M)
                Ms = fstar * self.halos.tab_M
                ssfr[i,:] = self.get_ssfr(z=z, Ms=Ms)

            self._tab_ssfr_ = ssfr

        return self._tab_ssfr_

    @property
    def tab_sfr(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.

        This does NOT set the SFR to zero in halos with M < Mmin or M > Mmax!
        Doing so screws up spline fitting in SFR...but if we don't need the
        SFR function for anything...shouldn't we just do it?

        """
        if not hasattr(self, '_tab_sfr_'):

            if self.pf['pop_sfr_model'] == 'sfr-func':
                self._tab_sfr_ = \
                    np.zeros((self.halos.tab_z.size, self.halos.tab_M.size))

                for i, z in enumerate(self.halos.tab_z):

                    if z > self.zform:
                        continue

                    if z < self.zdead:
                        continue

                    # Should be a little careful here: need to go one or two
                    # steps past edge to avoid interpolation problems in SFRD.

                    # SF fueld by accretion onto halos already above threshold
                    #if self.pf['pop_sfr_above_threshold']:

                    if self.pf['pop_sfr_model'] == 'sfr-func':
                        self._tab_sfr_[i] = self.get_sfr(z=z, Mh=self.halos.tab_M)
                    else:
                        raise ValueError('shouldnt happen.')
            elif self.pf['pop_sfr_model'] == 'sfr-tab':
                self._tab_sfr_ = self.pf['pop_sfr']
                assert self._tab_sfr_.shape == \
                    (self.halos.tab_z.size, self.halos.tab_M.size)
            elif self.pf['pop_sfr_model'] == '21cmfast':
                Mb = self.cosm.fbar_over_fcdm * self.halos.tab_M
                fst = self.tab_fstar
                tstar = self.pf['pop_tstar']
                H = self.cosm.HubbleParameter(self.halos.tab_z) * s_per_yr
                self._tab_sfr_ = np.array([Mb * fst[i] * H[i] / tstar \
                    for i in range(H.size)])
            elif self.pf['pop_sfr_model'] == 'smhm-func':
                Ms = self.tab_fstar * self.halos.tab_M
                self._tab_sfr_ = self.tab_ssfr * Ms
            elif self.is_quiescent:
                self._tab_sfr_ = self._tab_sfr_ = \
                    np.zeros((self.halos.tab_z.size, self.halos.tab_M.size))
            else:
                self._tab_sfr_ = self._tab_eta \
                    * self.cosm.fbar_over_fcdm \
                    * self.halos.tab_MAR * self.tab_fstar

            #self._tab_sfr_mask_ = np.zeros_like(self._tab_sfr_, dtype=bool)

            # Why am I getting a NaN?
            isnan = np.isnan(self._tab_sfr_)

            if isnan.sum() > 1 and self.pf['debug']:
                # Find bounds in redshift and mass?
                i_nan = np.argwhere(isnan==1)
                x, y = i_nan.T
                i_zlo = np.argmin(x)
                i_zhi = np.argmax(x)
                i_Mlo = np.argmin(y)
                i_Mhi = np.argmax(y)

                zlo = self.halos.tab_z[i_zlo]
                zhi = self.halos.tab_z[i_zhi]
                Mlo = self.halos.tab_M[i_Mlo]
                Mhi = self.halos.tab_M[i_Mhi]

                print("WARNING: {} Nans detected in _tab_sfr_".format(isnan.sum()))
                print("WARNING: Found in range {}<=z<={} and {}<=Mh<={}".format(zlo,
                    zhi, Mlo, Mhi))
                print("Note: pop_sfr_model={}".format(self.pf['pop_sfr_model']))

            self._tab_sfr_[isnan] = 0.
            self._tab_sfr_ *= ~self._tab_sfr_mask

        return self._tab_sfr_

    def get_nh_active(self, z):
        """
        Compute number of active halos at given redshift `z`.
        """
        if not hasattr(self, '_nactive'):
            self._func_nactive = interp1d(self.halos.tab_z, self.tab_nh_active,
                kind=self.pf['pop_interp_sfrd'])
        return self._func_nactive(z)

    @property
    def tab_nh_active(self):
        if not hasattr(self, '_tab_nh_active_'):
            self._tab_nh_active_ = np.ones_like(self.halos.tab_z)

            lnM = np.log(self.halos.tab_M)

            # Loop from high-z to low-z
            for k, z in enumerate(self.halos.tab_z[-1::-1]):

                i = self.halos.tab_z.size - k - 1

                #if not self.pf['pop_sfr_above_threshold']:
                #    break

                if z > self.zform:
                    continue

                integrand = self.halos.tab_dndlnm[i] * self.tab_focc[k,:]

                # Mmin and Mmax will never be exactly on Mh grid points
                # so we interpolate to more precisely determine SFRD.

                c1 = self.halos.tab_M >= self._tab_Mmin[i]
                c2 = self.halos.tab_M <= self._tab_Mmax[i]
                ok = np.logical_and(c1, c2)

                if self._tab_Mmin[i] == self._tab_Mmax[i]:
                    self._tab_nh_active_[i] = 0

                    # We 'break' here because once Mmax = Mmin, PopIII
                    # should be gone forever.

                    if z < self.zform:#self.pf['initial_redshift']:
                        break
                    else:
                        continue

                # This can happen if Mmin and Mmax are within the same mass bin,
                # or if they straddle a single mass bin. Need to be a bit careful.
                if ok.sum() in [0, 1]:
                    i1 = np.argmin(np.abs(self.halos.tab_M - self._tab_Mmin[i]))
                    if self.halos.tab_M[i1] > self._tab_Mmin[i]:
                        i1 -= 1
                    i2 = i1 + 1

                    # Trapezoid here we come
                    b = self._tab_logMmax[i] - self._tab_logMmin[i]

                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])

                    h = abs(y2 - y1)

                    self._tab_nh_active_[i] = 0.5 * b * (y1 + y2)

                    continue

                # Otherwise, do the normal thing.
                #if ok.sum() == 1:
                #    iok = [np.argwhere(ok).squeeze()]
                #else:
                iok = np.argwhere(ok).squeeze()

                Mlo1 = min(iok)
                Mhi1 = max(iok)
                Mlo2 = Mlo1 - 1
                Mhi2 = Mhi1 + 1

                # This happens if Mmin and Mmax straddle a mass bin
                if ok.sum() == 1:
                    raise ValueError('help')
                    b = np.log(self.halos.tab_M[Mlo1+1]) - np.log(self.halos.tab_M[Mlo1])
                    #h = abs(integrand[Mlo1+1] - integrand[Mlo1])
                    #b = self.halos.lnM[Mlo1] - self.self.halos.lnM[Mlo1+1]

                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])

                    h = abs(y2 - y1)

                    tot = 0.5 * b * h
                else:
                    # This is essentially an integral from Mlo1 to Mhi1
                    tot = np.trapz(integrand[ok], x=np.log(self.halos.tab_M[ok]))
                integ_lo = np.trapz(integrand[Mlo2:Mhi1+1],
                    x=np.log(self.halos.tab_M[Mlo2:Mhi1+1]))

                # Interpolating over lower integral bound
                sfrd_lo = np.interp(self._tab_logMmin[i],
                    [np.log(self.halos.tab_M[Mlo2]), np.log(self.halos.tab_M[Mlo1])],
                    [integ_lo, tot]) - tot

                if Mhi2 >= self.halos.tab_M.size:
                    sfrd_hi = 0.0
                else:
                    integ_hi = np.trapz(integrand[Mlo1:Mhi2+1],
                        x=np.log(self.halos.tab_M[Mlo1:Mhi2+1]))
                    sfrd_hi = np.interp(self._tab_logMmax[i],
                        [np.log(self.halos.tab_M[Mhi1]), np.log(self.halos.tab_M[Mhi2])],
                        [tot, integ_hi]) - tot

                self._tab_nh_active_[i] = tot + sfrd_lo + sfrd_hi

            self._tab_nh_active_ *= 1. / cm_per_mpc**3

        return self._tab_nh_active_

    @property
    def tab_sfrd_total(self):
        """
        SFRD as a function of redshift.

            ..note:: Units are Msun/yr/cMpc^3.

        """

        if not hasattr(self, '_tab_sfrd_total_'):
            Nz = self.halos.tab_z.size

            ok = ~self._tab_sfr_mask
            integrand = self.tab_sfr * self.halos.tab_dndlnm * self.tab_focc

            ##
            # Use cumtrapz instead and interpolate onto Mmin, Mmax
            ##
            ct = 0
            self._tab_sfrd_total_ = np.zeros_like(self.halos.tab_z)
            for _i, z in enumerate(self.halos.tab_z[-1::-1]):

                i = self.halos.tab_z.size - _i - 1

                if z > self.zform:
                    continue

                if z <= self.zdead:
                    break

                # See if Mmin and Mmax fall in the same bin, in which case
                # we'll just set SFRD -> 0 to avoid numerical nonsense.
                #j1 = np.argmin(np.abs(self._tab_Mmin[i] - self.halos.tab_M))
                #j2 = np.argmin(np.abs(self._tab_Mmax[i] - self.halos.tab_M))
                #if j1 == j2:
                #    if abs(self._tab_Mmax[i] / self._tab_Mmin[i] - 1) < 1e-2:
                #        continue

                tot = np.trapz(integrand[i], x=np.log(self.halos.tab_M))
                cumtot = cumtrapz(integrand[i], x=np.log(self.halos.tab_M),
                    initial=0.0)

                above_Mmin = np.interp(np.log(self._tab_Mmin[i]),
                        np.log(self.halos.tab_M), tot - cumtot)
                above_Mmax = np.interp(np.log(self._tab_Mmax[i]),
                        np.log(self.halos.tab_M), tot - cumtot)

                #if above_Mmin < above_Mmax:
                #    print("WARNING: SFRD(>Mmin) < SFRD(>Mmax) at z={}".format(z))

                self._tab_sfrd_total_[i] = above_Mmin - above_Mmax

                # Once the SFRD is zero, it's zero. Prevents annoying
                # numerical artifacts / wiggles / spikes at late times,
                # particularly when Mmin ~ Mmax.
                continue

                #now = self._tab_sfrd_total_[i]
                #pre = self._tab_sfrd_total_[i+1]
                #if now == 0 and pre > 0 and ct == 0:
                #    z_dead = z
                #    ct += 1
                #elif now == 0 and ct > 0:
                #    if (z_dead - z) >= 2 and abs(z - self.zdead) < 0.2:
                #        break

            #self._tab_sfrd_total_ *= g_per_msun / s_per_yr / cm_per_mpc**3

        return self._tab_sfrd_total_

    def get_sfrd_in_mag_range(self, z, lo=None, hi=-17, absolute=True, x=1600,
        band=None, units='Angstrom', window=1, load=True, raw=False, nebular_only=False,
        apply_dustcorr=False):
        """
        Return SFRD integrated above some limiting magnitude.

        .. note :: Relatively crude at this stage. No interpolation, just
            using nearest grid points in (z, Mh) space.

        Parameters
        ----------
        z : int, float
            Redshift.
        lo, hi : int, float
            Magnitude cuts of interest.
        absolute : bool
            If True, interpret `mag` as an absolute AB magnitude.

        Returns
        -------
        SFRD in units of Msun/yr/cMpc^3.
        """

        mags = self.get_mags(z, absolute=absolute, x=x, band=band, units=units,
            window=window, load=load, raw=raw,
            nebular_only=nebular_only)

        #Mh = self.get_mass(z, kind='halo')
        Mh = self.halos.tab_M

        if hi is not None:
            Mlo = Mh[np.argmin(np.abs(mags - hi))]
        else:
            Mlo = 0

        if lo is not None:
            Mhi = Mh[np.argmin(np.abs(mags - lo))]
        else:
            Mhi = None

        return self.get_sfrd_in_mass_range(z, Mlo=Mlo, Mhi=Mhi)

    def get_sfrd_in_mass_range(self, z, Mlo, Mhi=None):
        """
        Compute SFRD within given halo mass range, [Mlo, Mhi], each in Msun.

        .. note :: Relatively crude at this stage. No interpolation, just
            using nearest grid points in (z, Mh) space.

        Returns
        -------
        SFRD in units of Msun/yr/cMpc^3.
        """

        # Check for exact match
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        if abs(self.halos.tab_z[iz] - z) < ztol:
            exact_match = True
        else:
            exact_match = False
            print("* WARNING: requested `z` not in grid, no interpolation implemented yet!")

        ok = ~self._tab_sfr_mask
        integrand = ok * self.tab_sfr * self.halos.tab_dndlnm * self.tab_focc

        ilo = np.argmin(np.abs(self.halos.tab_M - Mlo))
        if Mhi is None:
            ihi = self.halos.tab_M.size
        else:
            ihi = np.argmin(np.abs(self.halos.tab_M - Mhi))

        _sfrd_tab = np.trapz(integrand[iz,ilo:ihi+1],
            x=np.log(self.halos.tab_M[ilo:ihi+1]))

        #_sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3

        return _sfrd_tab

    @property
    def tab_focc(self):
        if not hasattr(self, '_tab_focc_'):
            yy, xx = self._tab_Mz
            focc = self.get_focc(z=xx, Mh=yy)

            if type(focc) in [int, float, np.float64]:
                self._tab_focc_ = focc * np.ones_like(self.halos.tab_dndm)
            else:
                self._tab_focc_ = focc

            self._tab_focc_ = np.minimum(self._tab_focc_, 1)
            self._tab_focc_ = np.maximum(self._tab_focc_, 0)

        return self._tab_focc_

    @property
    def tab_fsurv(self):
        if not hasattr(self, '_tab_fsurv_'):
            yy, xx = self._tab_Mz
            fsurv = self.get_fsurv(z=xx, Mh=yy)

            if type(fsurv) in [int, float, np.float64]:
                self._tab_fsurv_ = fsurv * np.ones_like(self.halos.tab_dndm)
            else:
                self._tab_fsurv_ = fsurv

            if self.pf['pop_fsurv_inv']:
                self._tab_fsurv_ = 1. - self._tab_fsurv_

        return self._tab_fsurv_

    @tab_fsurv.setter
    def tab_fsurv(self, value):
        if self.pf['pop_fsurv_inv']:
            self._tab_fsurv_ = 1 - value
        else:
            self._tab_fsurv_ = value

    def get_smhm(self, **kwargs):
        if self.pf['pop_sfr_model'] in ['smhm-func']:
            return self.get_fstar(**kwargs)
        else:
            raise NotImplemented('help')

    def get_sfe(self, **kwargs):
        return self.get_fstar(**kwargs)

    def get_fstar(self, **kwargs):
        """
        Compute star formation efficiency (SFE).

        .. note :: Takes keyword arguments only (see below).


        Parameters
        ----------
        z : int, float
            Redshift
        Mh : int, float, np.ndarray
            Halo mass(es) in Msun.


        """

        if hasattr(self, '_get_fstar'):
            return self._get_fstar(**kwargs)

        if self.pf['pop_sfr_model'] in ['uvlf', 'ham']:
            if type(kwargs['z']) == np.ndarray:

                if hasattr(self, '_sfe_ham'):
                    return self._sfe_ham(kwargs['z'], kwargs['Mh'])

                #
                z = np.unique(kwargs['z'])
                Mh = kwargs['Mh']

                if Mh.ndim == 2:
                    fstar = np.zeros_like(Mh)
                else:
                    fstar = np.zeros((z.size, Mh.size))

                pb = ProgressBar(z.size, use=self.pf['progress_bar'],
                    name='ham')
                pb.start()
                for iz, _z_ in enumerate(z):
                    if Mh.ndim == 1:
                        fstar[iz,:] = self.run_abundance_match(z=_z_, Mh=Mh)
                    else:
                        fstar[:,iz] = self.run_abundance_match(z=_z_,
                            Mh=Mh[:,iz])

                    pb.update(iz)

                pb.finish()

                return fstar
            else:
                return self.run_abundance_match(z=kwargs['z'],
                    Mh=kwargs['Mh'])
        else:
            if not self.pf['pop_star_formation']:
                self._get_fstar = lambda **kwargs: 0.0
                return self._get_fstar(**kwargs)


            assert self.pf['pop_sfr'] is None

            ##
            # Some models based on mass-loading factor, convert to fstar
            if self.pf['pop_mlf'] is not None:
                _func = self._get_function('pop_mlf')

                func = lambda **kwargs: self.get_fshock(**kwargs) \
                    / ((1. / self.pf['pop_fstar_max'] + _func(**kwargs)))
            ##
            # fstar itself, easy.
            elif self.pf['pop_fstar'] is not None:
                func = self._get_function('pop_fstar')
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')

            # Boost factor is to do metallicity dependent effects without
            # having to adjust fstar parameters
            if self.pf['pop_calib_lum'] is not None:
                assert self.pf['pop_ssp'] == False
                wave = self.pf['pop_calib_wave']
                boost = self.pf['pop_calib_lum'] \
                    / self.src.get_lum_per_sfr(x=wave, units='Angstrom')
            else:
                boost = 1.

            _func_ = lambda **kwargs: func(**kwargs) * boost
            self._get_fstar = _func_

            return self._get_fstar(**kwargs)

    #@cached_property
    #def tab_yield_per_sfr(self):
    #    # Need this to avoid inheritance issue with GalaxyAggregate
    #    #if not hasattr(self, '_yield_per_sfr'):

    #    #if type(self.rad_yield) is FunctionType:
    #    #    self._yield_per_sfr = self.rad_yield()
    #    #else:
    #    #    self._yield_per_sfr = self.rad_yield

    #

    #    return self._yield_per_sfr


    @property
    def _tab_Mz(self):
        if not hasattr(self, '_tab_Mz_'):
            yy, xx = np.meshgrid(self.halos.tab_M, self.halos.tab_z)
            self._tab_Mz_ = yy, xx
        return self._tab_Mz_

    @property
    def tab_fstar(self):
        if not hasattr(self, '_tab_fstar_'):
            yy, xx = self._tab_Mz
            # Should be like tab_dndm
            if self.is_uvlf_parametric:
                self._tab_fstar_ = np.zeros_like(yy)
                for i, _z_ in enumerate(self.halos.tab_z):
                    self._tab_fstar_[i] = self.get_sfe(z=_z_, Mh=self.halos.tab_M)
            else:
                self._tab_fstar_ = self.get_sfe(z=xx, Mh=yy)

        return self._tab_fstar_

    def _SAM(self, z, y):
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z(z, y)
        else:
            raise NotImplementedError('No SAM with nz={}'.format(\
                self.pf['pop_sam_nz']))

    def _SAM_jac(self, z, y): # pragma: no cover
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z_jac(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z_jac(z, y)
        else:
            raise NotImplementedError('No SAM with nz={}'.format(\
                self.pf['pop_sam_nz']))

    def _SAM_1z(self, z, y):
        """
        Simple semi-analytic model for the components of galaxies.

        Really just the right-hand sides of a set of ODEs describing the
        rate of change in the halo mass, stellar mass, and metal mass.
        Other elements can be added quite easily.

        All terms have units of per year. The last equation is simply
        the evolution of time wrt redshift.

        Parameters
        ----------
        z : int, float
            current redshift
        y : array
            halo mass, gas mass, stellar mass, gas-phase metallicity

        Returns
        -------
        An updated array of y-values.

        """

        Mh, Mg, Mst, MZ, cMst, Mbh = y

        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg, 'MZ': MZ,
            'cMst': cMst, 'Mbh': Mbh}

        # Assume that MZ, Mg, and Mstell acquired *not* by smooth inflow
        # is same fraction of accreted mass as fractions in this halo
        # right now

        fb = self.cosm.fbar_over_fcdm

        # Convert from s/dz to yr/dz
        dtdz_s = -self.cosm.dtdz(z)
        dtdz = dtdz_s / s_per_yr

        # Splitting up the inflow. P = pristine.
        # Units = Msun / yr -> Msun / dz
        #if self.pf['pop_sfr_model'] in ['sfe-func']:
        PIR = fb * self.get_mar(z, Mh) * dtdz
        #NPIR = fb * self.get_MDR(z, Mh) * dtdz
        MGR = np.atleast_1d(self.MGR(z, Mh))
        #else:
        #    PIR = NPIR = MGR = 0.0

        # Measured relative to baryonic inflow
        Mb = fb * Mh
        Zfrac = self.pf['pop_acc_frac_metals'] * (MZ / Mb)
        Sfrac = self.pf['pop_acc_frac_stellar'] * (Mst / Mb)
        Gfrac = self.pf['pop_acc_frac_gas'] * (Mg / Mb)

        # Need SFR per dz
        if not self.pf['pop_star_formation']:
            fstar = SFR = 0.0
        elif self.pf['pop_sfr'] is None:
            fstar = self.get_sfe(**kw)
            SFR = PIR * fstar
        else:
            fstar = 1e-10
            SFR = self.get_sfr(**kw) * dtdz

        # "Quiet" mass growth
        #fsmooth = self.fsmooth(**kw)

        # Eq. 1: halo mass.
        y1p = MGR * dtdz

        # Eq. 2: gas mass
        if self.pf['pop_sfr'] is None:
            y2p = PIR * (1. - SFR/PIR) #+ NPIR * Gfrac
        else:
            y2p = PIR * (1. - fstar) #+ NPIR * Gfrac

        # Add option of parameterized stifling of gas supply, and
        # ejection of gas.

        if self._done_setting_Mmax:
            Mmax = self.get_Mmax(z)
        else:
            Mmax = np.inf

        # Eq. 3: stellar mass
        Mmin = self.get_Mmin(z)
        if (Mh < Mmin) or (Mh > Mmax):
            y3p = SFR = np.array([0.0])
        else:
            y3p = SFR * (1. - self.pf['pop_mass_yield']) #+ NPIR * Sfrac

        # Eq. 4: metal mass -- constant return per unit star formation for now
        if self.pf['pop_enrichment']:
            y4p = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield'] * SFR \
                * (1. - self.pf['pop_mass_loss']) #\
                #+ NPIR * Zfrac
        else:
            y4p = np.array([0.0])

        if (Mh < Mmin) or (Mh > Mmax):
            y5p = np.array([0.0])
        else:
            y5p = SFR #+ NPIR * Sfrac

        # BH accretion rate
        if self.pf['pop_bh_formation']:
            if self.pf['pop_bh_facc'] is not None:
                y6p = self.pf['pop_bh_facc'] * PIR
            else:

                eta = self.pf['pop_eta']
                fduty = self.pf['pop_fduty']
                if Mbh > 0:
                    y6p = Mbh * dtdz_s * fduty * (1. - eta) / eta / t_edd
                else:
                    y6p = np.array([0.0])

        else:
            y6p = np.array([0.0])

        # Stuff to add: parameterize metal yield, metal escape, star formation
        # from reservoir? How to deal with Mmin(z)? Initial conditions (from PopIII)?

        #print(y1p, y2p, y3p, y4p, y5p, y6p)
        results = [y1p, y2p, y3p, y4p, y5p, y6p]

        return np.array(results)

    def _SAM_1z_jac(self, z, y): # pragma: no cover
        """
        Jacobian for _SAM_1z
        """

        print('jac!', z, y)

        Mh, Mg, Mst, MZ, cMst, Mbh = y

        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg, 'MZ': MZ,
            'cMst': cMst, 'Mbh': Mbh}

        # Assume that MZ, Mg, and Mstell acquired *not* by smooth inflow
        # is same fraction of accreted mass as fractions in this halo
        # right now

        fb = self.cosm.fbar_over_fcdm

        # Convert from s/dz to yr/dz
        dtdz_s = -self.cosm.dtdz(z)
        dtdz = dtdz_s / s_per_yr

        # Splitting up the inflow. P = pristine.
        # Units = Msun / yr -> Msun / dz
        #if self.pf['pop_sfr_model'] in ['sfe-func']:
        PIR = fb * self.get_mar(z, Mh) * dtdz
        #NPIR = fb * self.get_MDR(z, Mh) * dtdz
        #else:
        #    PIR = NPIR = 0.0 # unused

        # Measured relative to baryonic inflow
        Mb = fb * Mh
        Zfrac = self.pf['pop_acc_frac_metals'] * (MZ / Mb)
        Sfrac = self.pf['pop_acc_frac_stellar'] * (Mst / Mb)
        Gfrac = self.pf['pop_acc_frac_gas'] * (Mg / Mb)

        # Need SFR per dz
        #if not self.pf['pop_star_formation']:
        #    fstar = SFR = 0.0
        #elif self.pf['pop_sfr'] is None:
        #    fstar = lambda _Mh: self.SFE(z=kw['z'], Mh=_Mh)
        #    SFR = lambda _Mh: PIR(_Mh) * fstar(_Mh)
        #else:
        #    fstar = 1e-10
        #    SFR = lambda _Mh: self.sfr(z=kw['z'], Mh=_Mh) * dtdz

        # Need SFR per dz
        if not self.pf['pop_star_formation']:
            fstar = SFR = 0.0
        elif self.pf['pop_sfr'] is None:
            fstar = self.get_sfe(**kw)
            SFR = PIR * fstar
        else:
            fstar = 1e-10
            SFR = self.get_sfr(**kw) * dtdz

        # "Quiet" mass growth
        #fsmooth = self.fsmooth(**kw)

        # Eq. 1: halo mass.
        _y1p = lambda _Mh: self.MGR(z, _Mh) * dtdz
        y1p = derivative(_y1p, Mh)

        # Eq. 2: gas mass
        if self.pf['pop_sfr'] is None:
            y2p = PIR * (1. - SFR/PIR) #+ NPIR * Gfrac
        else:
            y2p = PIR * (1. - fstar) #+ NPIR * Gfrac

        #_yp = lambda _Mh: self.MGR(z, _Mh) * dtdz
        #y2p = derivative(_yp2, Mh)

        # Add option of parameterized stifling of gas supply, and
        # ejection of gas.

        if self._done_setting_Mmax:
            Mmax = self.get_Mmax(z)
        else:
            Mmax = np.inf

        # Eq. 3: stellar mass
        Mmin = self.get_Mmin(z)
        if (Mh < Mmin) or (Mh > Mmax):
            y3p = SFR = 0.
        else:
            y3p = SFR * (1. - self.pf['pop_mass_yield']) + NPIR * Sfrac

        # Eq. 4: metal mass -- constant return per unit star formation for now
        y4p = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield'] * SFR \
            * (1. - self.pf['pop_mass_loss']) \
            + NPIR * Zfrac

        if (Mh < Mmin) or (Mh > Mmax):
            y5p = 0.
        else:
            y5p = SFR + NPIR * Sfrac

        # Add BHs
        if self.pf['pop_bh_formation']:
            if self.pf['pop_bh_facc'] is not None:
                y6p = self.pf['pop_bh_facc'] * PIR
            else:

                C = dtdz_s * 4.0 * np.pi * G * m_p / sigma_T / c

                if self.pf['pop_bh_seed_mass'] is not None:
                    Mseed = self.pf['pop_bh_seed_mass']
                elif self.pf['pop_bh_seed_eff'] is not None:
                    Mseed = self.pf['pop_bh_seed_eff'] * Mg
                else:
                    Mseed = self.pf['pop_bh_seed_ratio'] * Mmin

                # Form new BHs
                if (Mh >= Mmin) and (Mbh == 0.0):
                    y6p = Mseed * C
                elif Mbh > 0:
                    # Eddington-limited growth. Remember Mbh is really
                    # just the accreted mass so we need to add in the seed mass.
                    y6p = C * (Mbh + Mseed)
                else:
                    y6p = 0.0

        else:
            y6p = 0.0

        # Remember that we're building a matrix. Columns:
        # [Mh, Mg, Mst, MZ, cMst, Mbh]
        # First, for testing, just do diagonal elements.
        results = [y1p, 0.0, 0.0, 0.0, 0.0, 0.0]

        return np.diag(results)

    def _SAM_2z(self, z, y): # pragma: no cover
        raise NotImplemented('Super not done with this!')

        Mh, Mg_cgm, Mg_ism_c, Mg_ism_h, Mst, MZ_ism, MZ_cgm = y

        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg}

        #
        fstar = self.get_sfe(**kw)
        tstar = 1e7 * s_per_yr

        Mdot_h = -1. * self.get_mar(z, Mh) * self.cosm.dtdz(z) / s_per_yr

        # Need cooling curve here eventually.
        Z_cgm = MZ_cgm / Mg_cgm
        Mdot_precip = 0. # Mg_cgm

        # Eq. 1: halo mass.
        y1p = Mh_dot

        # Eq. 2: CGM mass
        # Pristine inflow + gas injected from galaxy winds / SN
        # - precipitation rate - losses from halo
        y2p = y1p * self.pf['pop_fstall']

        # Eq. 3: Hot ISM gas mass
        # Winds and SN, gas precipitated from CGM?
        y3p = y1p * (1. - self.pf['pop_fstall'])

        # Eq. 4: Cold ISM gas mass
        # Pristine inflow + hot ISM cooled off - star formation
        y4p = y1p * (1. - self.pf['pop_fstall'])

        # Eq. 5: Star formation
        Mmin = self.get_Mmin(z)
        if Mh < Mmin:
            y5p = 0.
        else:
            y5p = fstar * (self.cosm.fbar_over_fcdm * y1p * (1. - self.pf['pop_fstall']) \
                + (self.pf['pop_fstar_rec'] / fstar) * Mg_ism_c / tstar)

        # Eq. 4: metal mass -- constant return per unit star formation for now
        # Could make a PHP pretty easily.
        y6p = self.pf['pop_metal_yield'] * y3p * (1. - self.pf['pop_mass_loss'])

        results = [y1p, y2p, y3p, y4p]

        return np.array(results)

    def get_duration(self, zend=6.):
        """
        Calculate the duration of this population, i.e., time it takes to get
        from formation redshift to Mmax.
        """

        zform, zfin, Mfin, raw = self.MassAfter(M0=self.pf['pop_initial_Mh'])

        duration = []
        for i, zf in enumerate(zfin):
            duration.append(self.cosm.LookbackTime(zf, zform[i]) / s_per_yr / 1e6)

        duration = np.array(duration)

        # This is not quite what Rick typically plots -- his final masses are
        # those at z=6, and many of those halos will have been forming PopII
        # stars for awhile.

        # So, compute the final halo mass, including time spent after Mh > Mmax.
        Mend = []
        for i, z in enumerate(zform):

            new_data = self._sort_sam(z, zform, raw, sort_by='form')

            Mend.append(np.interp(zend, zform, new_data['Mh']))

        return zform, zfin, Mfin, duration#, np.array(Mend)

    def MassAfter(self, M0=0):
        """
        Compute the final mass of a halos that begin at Mmin and grow for dt.

        Parameters
        ----------
        dt : int, float
            Growth time [years]

        Returns
        -------
        Array of formation redshifts, final redshifts, and final masses.

        """

        # This loops over a bunch of formation redshifts
        # and computes the trajectories for all SAM fields.
        zarr, data = self.get_histories(M0=M0)

        # At this moment, all data is in order of ascending redshift
        # Each element in `data` is 2-D: (zform, zarr)

        # Figure out the final mass (after `dt`) of halos formed at each
        # formation redshift, and the redshift at which they reach that mass
        Mfin = []
        zfin = []
        for k, z in enumerate(zarr):

            # z here is the formation redshift
            new_data = self._sort_sam(z, zarr, data, sort_by='form')

            # Redshift when whatever transition-triggering event was reached
            zmax = new_data['zmax']
            zfin.append(zmax)

            # Interpolate in mass trajectory to find maximum mass
            # more precisely. Reminder: already interpolated to
            # find 'zmax' in call to _ScalingRelations
            Mmax = np.interp(zmax, zarr, new_data['Mh'])

            # We just want the final mass (before dt killed SAM)
            Mfin.append(Mmax)

        Mfin = self._apply_lim(Mfin, 'max', zarr)

        zfin = np.array(zfin)

        # Check for double-valued-ness
        # Just kludge and take largest value.

        zrev = zfin[-1::-1]
        for i, z in enumerate(zrev):
            if i == 0:
                continue

            if z == self.zdead:#self.pf['final_redshift']:
                break

        Nz = len(zfin)

        zfin[0:Nz-i] = self.zdead#self.pf['final_redshift']
        Mfin[0:Nz-i] = max(Mfin)


        return zarr, zfin, np.array(Mfin), data

    def _sort_sam(self, z, zarr, data, sort_by='obs'):
        """
        Take results of a SAM and grab data for a single formation redshift.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        zarr : np.ndarray
            Array of all redshifts used in SAM.
        data : dict


        Returns
        -------
        Dictionary containing trajectories
        """

        # First grab all elements with the right redshift.
        tmp = {}
        k = np.argmin(np.abs(z - zarr))
        for key in data.keys():
            if data[key].ndim == 2:
                if sort_by == 'form':
                    tmp[key] = data[key][k]
                else:
                    tmp[key] = data[key][:,k]
            else:
                tmp[key] = data[key][k]

        # Next, make sure they are in order of increasing halo mass
        if sort_by == 'form':
            new_data = tmp
        else:
            new_data = {}
            sorter = np.argsort(tmp['Mh'])
            for key in tmp.keys():
                if data[key].ndim == 2:
                    new_data[key] = tmp[key][sorter]
                else:
                    new_data[key] = tmp[key]

        return new_data

    @property
    def _trajectories(self):
        if not hasattr(self, '_trajectories_'):
            raise AttributeError('Must set by hand or run `Trajectories`.')
        return self._trajectories_

    @_trajectories.setter
    def _trajectories(self, value):
        self._trajectories_ = value

    @property
    def histories(self):
        if not hasattr(self, '_histories'):
            if self.pf['pop_histories'] is not None:
                if type(self.pf['pop_histories']) == tuple:
                    func, kw = self.pf['pop_histories']
                    self._histories = func(**kw)
                elif type(self.pf['pop_histories']) == dict:
                    self._histories = self.pf['pop_histories']
                else:
                    raise NotImplemented('help!')
            else:
                self._histories = self.get_histories()[1]
        return self._histories

    def get_histories(self, M0=0):
        """
        In this case, the formation time of a halo matters.

        Returns
        -------
        Dictionary of quantities, each having shape (z, z).

        The first dimension corresponds to formation time, the second axis
        represents trajectories. So, e.g., to pick out all halo masses at a
        given observed redshift (say z=6) you would do:

            zarr, data = self.get_histories()
            k = np.argmin(np.abs(zarr - 6))
            Mh = data[:,k]

        A check on this

        """

        if hasattr(self, '_trajectories'):
            return self._trajectories

        if not self.is_sam:
            raise NotImplementedError('This is an HOD model! No such thing as history.')

        keys = ['Mh', 'Mg', 'Ms', 'MZ', 'cMs', 'Mbh', 'SFR', 'SFE', 'MAR',
            'Md', 'Sd', 'nh', 'Z', 't']

        zf = max(float(self.halos.tab_z.min()), self.zdead)
        zi = min(float(self.halos.tab_z.max()), self.zform)

        if self.pf['sam_dz'] is not None:
            assert self.pf['hmf_dt'] is None
            dz = self.pf['sam_dz']
            zfreq = int(round(self.pf['sam_dz'] / dz, 0))
        else:
            zfreq = 1
            dz = np.diff(self.halos.tab_z)

        # Potential precision issues if zf is really zf+1e-8 or something
        # Correct `zarr` if first element is really equal to zf within
        # machine precision.
        in_range = np.logical_and(self.halos.tab_z > zf+small_dz,
            self.halos.tab_z <= zi)
        zarr = self.halos.tab_z[in_range][::zfreq]

        zmax = []
        zform = []

        if self.pf['halo_hist_Mmax'] is not None:
            dMmin = self.pf['halo_hist_dlogM']

            M0_aug = 10**np.arange(0+dMmin, np.log10(self.pf['halo_hist_Mmax'])+dMmin,
                dMmin)

            results = {key:np.zeros(((zarr.size+M0_aug.size, zarr.size))) \
                for key in keys}
        else:
            results = {key:np.zeros([zarr.size]*2) for key in keys}

        for i, z in enumerate(zarr):

            #if z == zarr[0]:
            #    continue
            #if (i == 0) or (i == len(zarr) - 1):
            #    zmax.append(zarr[i])
            #    zform.append(z)
            #    continue

            # If M0 is 0, assume it's the minimum mass at this redshift.
            _zarr, _results = self.run_sam(z0=z, M0=M0)

            # Need to splice into the right elements of 2-D array.
            # SAM is run from zform to final_redshift, so only a subset
            # of elements in the 2-D table are filled.
            for key in keys:
                dat = _results[key].copy()
                k = np.argmin(abs(_zarr.min() - zarr))
                results[key][i,k:k+len(dat)] = dat.squeeze()

            zform.append(z)

            zmax.append(_results['zmax'])

        ##
        # Fill-in high-mass end?
        # Because we "launch" halos at Mmin(z), we may miss out on very
        # high-mass halos at late times if we set Tmin very small, and of
        # course we'll miss out on the early histories of small halos if
        # Tmin is large. So, fill in histories by incrementing above Mmin
        # at highest available redshsift.
        if self.pf['halo_hist_Mmax'] is not None:

            _z0 = zarr.max()
            i0 = zarr.size

            if self.pf['verbose']:
                print("# Augmenting suite of halos at z_form={:.2f}".format(_z0))
                print("# Will generate halos with M0 up to M0={:.1f}*Mmin".format(
                    M0_aug.max()))

            for i, _M0 in enumerate(M0_aug):
                # If M0 is 0, assume it's the minimum mass at this redshift.
                _zarr, _results = self.run_sam(z0=_z0, M0=_M0)

                # Need to splice into the right elements of 2-D array.
                # SAM is run from zform to final_redshift, so only a subset
                # of elements in the 2-D table are filled.
                for key in keys:
                    dat = _results[key].copy()
                    k = np.argmin(abs(_zarr.min() - zarr))
                    results[key][i0+i,k:k+len(dat)] = dat.squeeze()

                zform.append(z)
                zmax.append(_results['zmax'])

        ##
        # Array-ify results
        results['zmax'] = np.array(zmax)
        results['zform'] = np.array(zform)
        results['z'] = zarr

        self._trajectories = np.array(zform), results

        return np.array(zform), results

    def run_sam(self, z0=None, M0=0):
        """
        Evolve a halo from initial mass M0 at redshift z0 forward in time.

        .. note :: If M0 is not supplied, we'll assume it's Mmin(z0).

        Parameters
        ----------
        z0 : int, float
            Formation redshift.
        M0 : int, float
            Formation mass (total halo mass).

        Returns
        -------
        redshifts, halo mass, gas mass, stellar mass, metal mass

        """

        # jac=self._SAM_jac
        solver = ode(self._SAM).set_integrator('lsoda',
            nsteps=1e4, atol=self.pf['sam_atol'], rtol=self.pf['sam_rtol'],
            with_jacobian=False)

        # Criteria used to kill a population.
        has_e_limit = self.pf['pop_bind_limit'] is not None
        has_T_limit = self.pf['pop_temp_limit'] is not None
        has_t_limit = self.pf['pop_time_limit'] is not None
        has_m_limit = self.pf['pop_mass_limit'] is not None
        has_a_limit = self.pf['pop_abun_limit'] is not None

        has_t_ceil = self.pf['pop_time_ceil'] is not None

        if self.pf['pop_time_limit'] == 0:
            has_t_limit = False
        if self.pf['pop_bind_limit'] == 0:
            has_e_limit = False

        ##
        # Outputs have shape (z, z)
        ##
        n0 = 0.0

        # Our results don't depend on this, unless SFE depends on z
        if (z0 is None) and (M0 == 0):
            z0 = self.halos.tab_z.max()
            M0 = self._tab_Mmin[-1]
            raise NotImplemented('Is this used anymore?')
        elif (M0 <= 1):

            # If we're treating a continuum of halos.
            M0 = np.interp(z0, self.halos.tab_z, self._tab_Mmin)

            iz = np.argmin(np.abs(z0 - self.halos.tab_z))

            if np.allclose(z0, self.halos.tab_z[iz], rtol=1e-2):
                n0 = self._tab_n_Mmin[iz]
            else:
                print('hay problemas!', z0, self.halos.tab_z[iz])
        elif (M0 > 1):
            M0 = np.interp(z0, self.halos.tab_z, M0 * self._tab_Mmin)

            dM = self.pf['halo_hist_dlogM']

            # Set number density of these guys.
            _marr_ = np.arange(np.log10(M0) - 3 * dM, np.log10(M0) + 3 * dM,
                dM * 0.2)
            _ngtm = [self._spline_ngtm(z0, _m_) for _m_ in _marr_]
            func = interp1d(_marr_, _ngtm, kind='cubic')
            n0 = func(np.log10(M0)) - func(np.log10(M0) + dM)

        # Setup time-stepping
        zf = max(float(self.halos.tab_z.min()), self.zdead)

        in_range = np.logical_and(self.halos.tab_z > zf+small_dz,
            self.halos.tab_z <= z0)
        in_range2 = np.logical_and(self.halos.tab_z >= zf,
            self.halos.tab_z <= z0)
        if self.pf['sam_dz'] is not None:
            assert self.pf['hmf_dt'] is None
            dz = self.pf['sam_dz'] * np.ones_like(self.halos.tab_z)
            zfreq = int(round(self.pf['sam_dz'] / dz[0], 0))
        else:
            # Need to use different range to make sure we get at least one
            # element in `dz`
            dz = np.diff(self.halos.tab_z[in_range2])
            zfreq = 1

        zarr = self.halos.tab_z[in_range][::zfreq]
        Nz = zarr.size
        zrev = zarr[-1::-1]
        dzrev = dz[-1::-1]

        # Boundary conditions (pristine halo)
        Mg0 = self.cosm.fbar_over_fcdm * M0
        MZ0 = 0.0
        Mst0 = 0.0
        Mbh0 = 0.0
        seeded = False

        # Initial stellar mass -> 0, initial halo mass -> Mmin
        solver.set_initial_value(np.array([M0, Mg0, Mst0, MZ0, Mst0, Mbh0]), z0)

        # Only used occasionally
        zmax = None
        zmax_t = None
        zmax_m = None
        zmax_a = None
        zmax_T = None
        zmax_e = None

        # zmax really means the highest redshift where a certain
        # transition criterion is satisfied.

        Mh_t = []
        Mg_t = []
        Mst_t = []
        cMst_t = []
        Mbh_t = []
        sfr_t = []
        sfe_t = []
        mar_t = []
        nh_t = []
        metals = []
        lbtime = []
        Ehist = []
        redshifts = []
        for i in range(Nz):

            if dz.size == 0:
                break

            #if zarr[-1::-1][i] < zf:
            #    tmp = list(-99999 * np.ones_like(zarr[-1::-1][i:]))
            #    redshifts.extend(tmp)
            #    Mh_t.extend(tmp)
            #    Mg_t.extend(tmp)
            #    Mst_t.extend(tmp)
            #    metals.extend(tmp)
            #    cMst_t.extend(tmp)
            #    sfr_t.extend(tmp)
            #    mar_t.extend(tmp)
            #    nh_t.extend(tmp)
            #
            #    break

            # In descending order
            redshifts.append(zrev[i])
            Mh_t.append(solver.y[0])
            Mg_t.append(solver.y[1])
            Mst_t.append(solver.y[2])
            metals.append(solver.y[3])
            cMst_t.append(solver.y[4])
            sfr_t.append(self.get_sfr(z=redshifts[-1], Mh=Mh_t[-1]))
            nh_t.append(n0)

            if self.pf['pop_sfr_model'] in ['sfe-func']:
                mar_t.append(self.MGR(redshifts[-1], Mh_t[-1]))
            else:
                mar_t.append(0.0)

            Mmin = np.interp(redshifts[-1], self.halos.tab_z, self._tab_Mmin)

            if self.pf['pop_bh_seed_mass'] is not None:
                Mseed = self.pf['pop_bh_seed_mass']
            elif self.pf['pop_bh_seed_eff'] is not None:
                Mseed = self.pf['pop_bh_seed_eff'] * Mg
            else:
                Mseed = self.pf['pop_bh_seed_ratio'] * Mmin

            # Form new BHs
            if (Mh_t[-1] >= Mmin) and (not seeded):
                Mbh_t.append(Mseed)
                # Update solver position.
                pos = np.array([Mh_t[-1], Mg_t[-1], Mst_t[-1], metals[-1], cMst_t[-1], Mseed])
                solver.set_initial_value(pos, redshifts[-1])
                seeded = True
            elif (not seeded):
                Mbh_t.append(0.0)
            else:
                Mbh_t.append(solver.y[5])

            if 'sfe' in self.pf['pop_sfr_model']:
                sfe_t.append(self.get_sfe(z=redshifts[-1], Mh=Mh_t[-1]))

            z = zrev[i]

            lbtime_myr = self.cosm.LookbackTime(z, z0) \
                / s_per_yr / 1e6

            lbtime.append(lbtime_myr)

            # t_ceil is a trump card.
            # For example, in some cases the critical metallicity will never
            # be met due to high inflow rates.
            if has_t_limit or has_t_ceil:
                if has_t_limit:
                    tlim = self.time_limit(z=z, Mh=M0)
                elif has_t_ceil:
                    tlim = self.time_ceil(z=z, Mh=M0)

                if lbtime_myr >= tlim:
                    hit_dt = True

                    lbtime_myr_prev = self.cosm.LookbackTime(redshifts[-2], z0) \
                        / s_per_yr / 1e6

                    zmax_t = np.interp(tlim,
                        [lbtime_myr_prev, lbtime_myr], redshifts[-2:])

            if has_m_limit:
                Mnow = solver.y[2]

                if Mnow >= self.mass_limit(z=z, Mh=M0) and (zmax_m is None):
                    zmax_m = np.interp(self.mass_limit(z=z, Mh=M0), cMst_t[-2:],
                        redshifts[-2:])

            if has_a_limit and (zmax_a is None):

                # Subtract off metals accrued before crossing Eb limit?
                #if (zmax_e is not None) and self.pf['pop_lose_metals'] and i > 0:
                #    MZ_e = np.interp(zmax_e, redshifts[-1::-1], metals[-1::-1])
                #    Mg_e = np.interp(zmax_e, redshifts[-1::-1], Mg_t[-1::-1])
                #
                #    Zpre = (metals[-2] - MZ_e) / solver.y[1]
                #    Znow = (solver.y[3] - MZ_e) / solver.y[1]
                #elif self.pf['pop_lose_metals']:
                #    Zpre = Znow = 0.0
                #else:

                Znow = solver.y[3] / solver.y[1]
                if Znow >= self.abun_limit(z=z, Mh=M0) and i > 0:
                    Zpre = metals[-2] / Mg_t[-2]
                    Z_t = [Zpre, Znow]
                    zmax_a = np.interp(self.abun_limit(z=z, Mh=M0), Z_t,
                        redshifts[-2:])

            # These next two are different because the condition might
            # be satisfied *at the formation time*, which cannot (by definition)
            # occur for time or mass-limited sources.
            if has_T_limit:
                Mtemp = self.halos.VirialMass(z, self.pf['pop_temp_limit'])

                if solver.y[0] >= Mtemp:
                    zmax_T = np.interp(Mtemp, Mh_t[-2:], redshifts[-2:])

            if has_e_limit and (zmax_e is None):

                Eblim = self.pf['pop_bind_limit']
                Ebnow = self.halos.BindingEnergy(redshifts[-1], Mh_t[-1])
                Ehist.append(Ebnow)

                if (Ebnow >= Eblim):

                    # i == 0 means we satisfied this criterion at the
                    # formation redshift.
                    if i == 0:
                        zmax_e = z0
                    else:
                        zmax_e = np.interp(Eblim, Ehist[-2:], redshifts[-2:])

                    # Potentially require a halo to keep growing
                    # for pop_time_limit *after* crossing this barrier.
                    if has_t_limit and self.pf['pop_time_limit_delay']:
                        tlim = self.time_limit(z=z, Mh=M0)

                        lbtime_myr = self.cosm.LookbackTime(z, zmax_e) \
                            / s_per_yr / 1e6

                        if lbtime_myr >= tlim:
                            hit_dt = True

                            lbtime_myr_prev = self.cosm.LookbackTime(redshifts[-2], z0) \
                                / s_per_yr / 1e6

                            zmax_e = np.interp(tlim,
                                [lbtime_myr_prev, lbtime_myr], redshifts[-2:])

            # Once zmax is set, keep solving the rate equations but don't adjust
            # zmax.
            if zmax is None:

                # If binding energy or Virial temperature are a limiter
                if ((zmax_e is not None) and has_e_limit) or \
                   ((zmax_T is not None) and has_T_limit):

                    # Only transition if time/mass/Z is ALSO satisfied
                    if (self.pf['pop_limit_logic'] == 'and') and \
                       (has_t_limit or has_m_limit or has_a_limit):

                        if (zmax_t is not None):
                            zmax = zmax_t
                        if (zmax_m is not None):
                            zmax = zmax_m
                        if (zmax_a is not None):
                            zmax = zmax_a

                        # Take the *lowest* redshift.
                        if zmax is not None:
                            if has_e_limit:
                                zmax = min(zmax, zmax_e)
                            else:
                                zmax = min(zmax, zmax_T)

                    else:
                        zmax = zmax_e if has_e_limit else zmax_T

                # If no binding or temperature arguments, use time or mass
                if not (has_e_limit or has_T_limit):
                    if (zmax_t is not None):
                        zmax = zmax_t
                    if (zmax_m is not None):
                        zmax = zmax_m
                    if (zmax_a is not None):
                        zmax = zmax_a

                # play the trump card
                if has_t_ceil and (not has_t_limit):
                    zmax = max(zmax, zmax_t)

            #print(i, Nz, zarr[-1::-1][i], solver.t, dz[-1::-1][i], solver.t - dz[-1::-1][i])
            solver.integrate(solver.t-dzrev[i])

        if zmax is None:
            zmax = self.zdead

        # Everything will be returned in order of ascending redshift,
        # which will mean masses are (probably) declining from 0:-1
        z = np.array(redshifts)[-1::-1]
        Mh = np.array(Mh_t)[-1::-1]
        Mg = np.array(Mg_t)[-1::-1]
        Ms = np.array(Mst_t)[-1::-1]
        MZ = np.array(metals)[-1::-1]

        if self.pf['pop_dust_yield'] is not None:
            Md = self.get_dust_yield(z=z, Mh=Mh) * MZ
            Rd = self.get_dust_scale(z=z, Mh=Mh)
            # Assumes spherical symmetry, uniform dust density
            Sd = 3. * Md * g_per_msun / 4. / np.pi / (Rd * cm_per_kpc)**2
        else:
            Md = Rd = Sd = np.zeros_like(Mh)

        #f self.pf['pop_dust_yield'] > 0:
        #   tau = self.get_dust_absorption_coeff(wave=1600.)
        #lse:
        #   tau = None

        cMs = np.array(cMst_t)[-1::-1]
        Mbh = np.array(Mbh_t)[-1::-1]
        SFR = np.array(sfr_t)[-1::-1]
        SFE = np.array(sfe_t)[-1::-1]
        MAR = np.array(mar_t)[-1::-1]
        nh = np.array(nh_t)[-1::-1]
        tlb = np.array(lbtime)[-1::-1]

        # Derived
        results = {'Mh': Mh, 'Mg': Mg, 'Ms': Ms, 'MZ': MZ, 'Md': Md, 'cMs': cMs,
            'Mbh': Mbh, 'SFR': SFR, 'SFE': SFE, 'MAR': MAR, 'nh': nh,
            'Sd': Sd, 'zmax': zmax, 't': tlb}
        results['Z'] = self.pf['pop_metal_retention'] \
            * (results['MZ'] / results['Mg'])

        for key in results:
            results[key] = np.maximum(results[key], 0.0)

        return z, results

    def get_luminosity_density(self, z, x=None, band=None):
        """
        Return the integrated luminosity density in the (Emin, Emax) band.

        Parameters
        ----------
        z : int, flot
            Redshift of interest.

        Returns
        -------
        Luminosity density in erg / s / c-cm**3.

        """

        return self.get_emissivity(z, x=x, band=band)

    def get_photon_density(self, z, Emin=None, Emax=None):
        """
        Return the photon luminosity density in the (Emin, Emax) band.

        Parameters
        ----------
        z : int, flot
            Redshift of interest.

        Returns
        -------
        Photon luminosity density in photons / s / c-cm**3.

        """

        # erg / s / cm**3
        if self.is_emissivity_scalable:
            rhoL = self.get_emissivity(z, E=None, Emin=Emin, Emax=Emax)
            erg_per_phot = self._get_energy_per_photon(Emin, Emax) * erg_per_ev

            return rhoL / erg_per_phot
        else:
            return self._get_photon_density(z, Emin, Emax)

    def get_zeta(self, z):
        """
        Compute the ionizing efficiency.

        .. note :: This only ever gets used to calculate the bubble size
            distribution using the excursion set approach. There, it really only
            gets used to translate halo masses to bubble masses.

        Parameters
        ----------
        z: int, float
            Redshift of interest.

        Returns
        -------
        Tuple containing (halo masses, zeta for each halo).

        """

        Mh = self.halos.tab_M
        Mst_c = self.get_mass(z, kind='stellar_cumulative') # tab_M by default
        fstar = Mst_c / Mh

        const = self.cosm.b_per_g * m_H / self.cosm.fbaryon
        zeta = const * fstar * self.src.get_Nion() * self.get_fesc_UV(z=z, Mh=Mh)

        return Mh, zeta

    def _profile_delta(self, k, M, z):
        """
        Delta-function profile for the delta component of power spectrum (discrete galaxies)
        """
        return 1. * k**0

    def get_ps_shot(self, z, k, wave1=1600., wave2=1600., raw=False,
        nebular_only=False):
        """
        Return shot noise term of halo power spectrum.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        k : int, float, np.ndarray
            Wave-numbers of interests [1 / cMpc].
        wave1 : int, float
            Rest wavelength of interest [Angstrom]
        wave2 : int, float
            Rest wavelength of interest [Angstrom]

        Returns
        -------
        P(k)
        """

        if not self.pf['pop_include_shot']:
            return np.zeros_like(k)

        band1 = wave1 if type(wave1) not in numeric_types else None
        band2 = wave2 if type(wave2) not in numeric_types else None

        lum1 = self.get_lum(z, x=wave1, band=band1, units='Angstrom',
            raw=raw, nebular_only=nebular_only)
        lum2 = self.get_lum(z, x=wave1, band=band2, units='Angstrom',
            raw=raw, nebular_only=nebular_only)

        focc1 = focc2 = self.get_focc(z=z, Mh=self.halos.tab_M)

        ps = self.halos.get_ps_shot(z, k=k,
            lum1=lum1, lum2=lum2,
            mmin1=None, mmin2=None, focc1=focc1, focc2=focc2, ztol=1e-3)

        return ps

    def _cache_ps_2h(self, z, k, wave1, wave2, raw, nebular_only):
        if not hasattr(self, '_cache_ps_2h_'):
            self._cache_ps_2h_ = {}

        if (z, wave1, wave2, raw, nebular_only) in self._cache_ps_2h_:

            _k_, _ps_ = self._cache_ps_2h_[(z, wave1, wave2, raw, nebular_only)]

            print("# Note: interpolating cached power spectrum at z={}".format(z))
            ps = np.exp(np.interp(np.log(k), np.log(_k_), np.log(_ps_)))

            return ps

        return None

    def _cache_ps_1h(self, z, k, wave1, wave2, raw, nebular_only, prof):
        if not hasattr(self, '_cache_ps_1h_'):
            self._cache_ps_1h_ = {}

        if (z, wave1, wave2, raw, nebular_only, prof) in self._cache_ps_1h_:

            _k_, _ps_ = self._cache_ps_1h_[(z, wave1, wave2, raw, nebular_only, prof)]

            print("# Note: interpolating cached power spectrum at z={}".format(z))
            ps = np.exp(np.interp(np.log(k), np.log(_k_), np.log(_ps_)))

            return ps

    def get_ps_2h(self, z, k, wave1=1600., wave2=1600., raw=False, nebular_only=False):
        """
        Return 2-halo term of 3-d power spectrum.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        k : int, float, np.ndarray
            Wave-numbers of interests [1 / cMpc].
        wave1 : int, float
            Rest wavelength of interest [Angstrom]
        wave2 : int, float
            Rest wavelength of interest [Angstrom]

        Returns
        -------
        P(k)

        """

        if not self.pf['pop_include_2h']:
            return np.zeros_like(k)

        cached_result = self._cache_ps_2h(z, k, wave1, wave2, raw, nebular_only)
        if cached_result is not None:
            return cached_result

        prof = self._profile_delta

        # If `wave` is a number, this will have units of erg/s/Hz.
        # If `wave` is a tuple, this will just be in erg/s.
        if np.all(np.array(wave1) <= 912):
            lum1 = 0
        else:
            band = wave1 if type(wave1) not in numeric_types else None
            lum1 = self.get_lum(z, x=wave1, raw=raw,
                band=band, units='Angstrom',
                nebular_only=nebular_only)

        if np.all(np.array(wave2) <= 912):
            lum2 = 0
        else:
            band = wave2 if type(wave2) not in numeric_types else None
            lum2 = self.get_lum(z, x=wave2, raw=raw,
                band=band, units='Angstrom',
                nebular_only=nebular_only)

        focc1 = focc2 = self.get_focc(z=z, Mh=self.halos.tab_M)

        ps = self.halos.get_ps_2h(z, k=k, prof1=prof, prof2=prof,
            lum1=lum1, lum2=lum2,
            mmin1=None, mmin2=None, focc1=focc1, focc2=focc2, ztol=1e-3)

        #if type(k) is np.ndarray:
        #    self._cache_ps_2h_[(z, wave1, wave2, raw, nebular_only)] = k, ps

        return ps

    def get_prof(self, prof=None):
        """
        Set up a function for Fourier-transformed profile.

        Parameters
        ----------
        prof : None, str
            If provided, currently must be one of the following:
                'nfw', 'isl', 'isl_exp', 'exp', or 'delta'

            If not provided here, but the self.pf['pop_prof_1h'] is not None,
            the latter will be used.

        Returns
        -------
        A function of z, Mh, and k.

        """
        # Defer to user-supplied parameter if given
        if prof is None:
            if self.pf['pop_prof_1h'] is not None:
                prof = self.pf['pop_prof_1h']

        if prof in [None, 'nfw']:
            # Try to load lookup table.
            prof = self.halos.tab_u_nfw
            if prof is None:
                prof = self.halos.get_u_nfw
        elif prof == 'delta':
            prof = self._profile_delta
        elif prof == 'isl':
            prof = lambda zz, mm, kk: self.halos.get_u_isl(zz, mm, kk)
        elif prof == 'isl_exp':
            prof = lambda zz, mm, kk: self.halos.get_u_isl_exp(zz, mm, kk)
        elif prof == 'exp':
            prof = lambda zz, mm, kk: self.halos.get_u_isl(zz, mm, kk)
        elif prof == 'cgm_rahmati':
            prof = lambda zz, mm, kk: self.halos.get_u_cgm_rahmati(zz, mm, kk)
        elif prof == 'cgm_steidel':
            prof = lambda zz, mm, kk: self.halos.get_u_cgm_steidel(zz, mm, kk)
        else:
            raise NotImplementedError('Unrecognized `prof` option: {}'.format(
                prof
            ))

        return prof

    def get_ps_1h(self, z, k, wave1=1600., wave2=1600., raw=False,
        nebular_only=False, prof=None):
        """
        Return 1-halo term of 3-d power spectrum.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        k : int, float, np.ndarray
            Wave-numbers of interest [1 / cMpc].
        wave1 : int, float
            Rest wavelength of interest [Angstrom]
        wave2 : int, float
            Rest wavelength of interest [Angstrom]

        Returns
        -------
        P(k)
        """

        if not self.pf['pop_include_1h']:
            return np.zeros_like(k)

        cached_result = self._cache_ps_1h(z, k, wave1, wave2, raw, nebular_only, prof)
        #if cached_result is not None:
        #    print("Loading from cache")
        #    return cached_result

        # Default to NFW
        prof = self.get_prof(prof)

        # If `wave` is a number, this will have units of erg/s/Hz.
        # If `wave` is a tuple, this will just be in erg/s.
        if np.all(np.array(wave1) <= 912):
            lum1 = 0
        else:
            band = wave1 if type(wave1) not in numeric_types else None
            lum1 = self.get_lum(z, wave=wave1, raw=raw,
                band=band, band_units='Angstrom',
                nebular_only=nebular_only)
        if np.all(np.array(wave2) <= 912):
            lum2 = 0
        else:
            band = wave2 if type(wave2) not in numeric_types else None
            lum2 = self.get_lum(z, wave=wave2, raw=raw,
                band=band, band_units='Angstrom',
                nebular_only=nebular_only)

        focc1 = focc2 = self.get_focc(z=z, Mh=self.halos.tab_M)

        ps = self.halos.get_ps_1h(z, k=k, prof1=prof, prof2=prof, lum1=lum1,
            lum2=lum2, mmin1=None, mmin2=None, focc1=focc1, focc2=focc2,
            ztol=1e-3)

        if type(k) is np.ndarray:
            self._cache_ps_1h_[(z, wave1, wave2, raw, nebular_only, prof)] = k, ps

        return ps

    def get_ps_obs(self, scale, wave_obs1, wave_obs2=None, include_shot=True,
        include_1h=True, include_2h=True, scale_units='arcsec', use_pb=True,
        time_res=1, raw=False, nebular_only=False, prof=None):
        """
        Compute the angular power spectrum of this galaxy population.

        .. note :: This function uses the Limber (1953) approximation.

        Parameters
        ----------
        scale : int, float, np.ndarray
            Angular scale [scale_units]
        wave_obs : int, float, tuple
            Observed wavelength of interest [microns]. If tuple, will
            assume elements define the edges of a spectral channel.
        scale_units : str
            So far, allowed to be 'arcsec' or 'arcmin'.
        time_res : int
            Can degrade native time or redshift resolution by this
            factor to speed-up integral. Do so at your own peril. By
            default, will sample time/redshift integrand at native
            resolution (set by `hmf_dz` or `hmf_dt`).

        """

        _zarr = self.halos.tab_z
        _zok  = np.logical_and(_zarr > self.zdead, _zarr <= self.zform)
        zarr  = self.halos.tab_z[_zok==1]

        # Degrade native time resolution by factor of `time_res`
        if time_res != 1:
            zarr = zarr[::time_res]

        dtdz = self.cosm.dtdz(zarr)

        if wave_obs2 is None:
            wave_obs2 = wave_obs1

        name = '{:.2f} micron'.format(np.mean(wave_obs1)) if np.all(wave_obs1 == wave_obs2) \
            else '({:.2f} x {:.2f}) microns'.format(np.mean(wave_obs1),
            np.mean(wave_obs2))

        ##
        # Loop over scales of interest if given an array.
        if type(scale) is np.ndarray:

            assert type(scale[0]) in numeric_types

            ps = np.zeros_like(scale)

            pb = ProgressBar(scale.shape[0],
                use=use_pb and self.pf['progress_bar'],
                name=f'p(k,{name})')
            pb.start()

            for h, _scale_ in enumerate(scale):

                integrand = np.zeros_like(zarr)
                for i, z in enumerate(zarr):
                    if z < self.zdead:
                        continue
                    if z > self.zform:
                        continue

                    integrand[i] = self._get_ps_obs(z, _scale_,
                        wave_obs1, wave_obs2,
                        include_shot=include_shot,
                        include_1h=include_1h, include_2h=include_2h,
                        scale_units=scale_units, raw=raw,
                        nebular_only=nebular_only, prof=prof)

                ps[h] = np.trapz(integrand * zarr, x=np.log(zarr))

                pb.update(h)

            pb.finish()

        # Otherwise, just compute PS at a single k.
        else:

            integrand = np.zeros_like(zarr)
            for i, z in enumerate(zarr):
                integrand[i] = self._get_ps_obs(z, scale, wave_obs1, wave_obs2,
                    include_shot=include_shot, include_2h=include_2h,
                    scale_units=scale_units, raw=raw,
                    nebular_only=nebular_only, prof=prof)

            ps = np.trapz(integrand * zarr, x=np.log(zarr))

        ##
        # Extra factor of nu^2 to eliminate Hz^{-1} units for
        # monochromatic PS
        assert type(wave_obs1) == type(wave_obs2)

        if type(wave_obs1) in numeric_types:
            ps = ps * (c / (wave_obs1 * 1e-4)) * (c / (wave_obs2 * 1e-4))
        else:
            nu1 = c / (np.array(wave_obs1) * 1e-4)
            nu2 = c / (np.array(wave_obs2) * 1e-4)
            dnu1 = -np.diff(nu1)
            dnu2 = -np.diff(nu2)

            ps = ps * np.mean(nu1) * np.mean(nu2) / dnu1 / dnu2

        return ps

    def _get_ps_obs(self, z, scale, wave_obs1, wave_obs2, include_shot=True,
        include_1h=True, include_2h=True, scale_units='arcsec', raw=False,
        nebular_only=False, prof=None):
        """
        Compute integrand of angular power spectrum integral.
        """

        if wave_obs2 is None:
            wave_obs2 = wave_obs1

        ##
        # Convert to Angstroms in rest frame. Determine emissivity.
        # Note: the units of the emisssivity will be different if `wave_obs`
        # is a tuple vs. a number. In the former case, it will simply
        # be in erg/s/cMpc^3, while in the latter, it will carry an extra
        # factor of Hz^-1.
        if type(wave_obs1) in [int, float, np.float64]:
            is_band_int = False

            # Get rest wavelength in Angstroms
            wave1 = wave_obs1 * 1e4 / (1. + z)
            # Convert to photon energy since that what we work with internally
            E1 = h_p * c / (wave1 * 1e-8) / erg_per_ev
            nu1 = c / (wave1 * 1e-8)

            # [enu] = erg/s/cm^3/Hz
            enu1 = self.get_emissivity(z, E=E1) * ev_per_hz
            # Not clear about * nu at the end
        else:
            is_band_int = True

            # Get rest wavelengths
            wave1 = tuple(np.array(wave_obs1) * 1e4 / (1. + z))

            # Convert to photon energies since that what we work with internally
            E11 = h_p * c / (wave1[0] * 1e-8) / erg_per_ev
            E21 = h_p * c / (wave1[1] * 1e-8) / erg_per_ev

            # [enu] = erg/s/cm^3
            enu1 = self.get_emissivity(z, Emin=E21, Emax=E11)

        if type(wave_obs2) in [int, float, np.float64]:
            is_band_int = False

            # Get rest wavelength in Angstroms
            wave2 = wave_obs2 * 1e4 / (1. + z)
            # Convert to photon energy since that what we work with internally
            E2 = h_p * c / (wave2 * 1e-8) / erg_per_ev
            nu2 = c / (wave2 * 1e-8)

            # [enu] = erg/s/cm^3/Hz
            enu2 = self.get_emissivity(z, E=E2) * ev_per_hz
            # Not clear about * nu at the end
        else:
            is_band_int = True

            # Get rest wavelengths
            wave2 = tuple(np.array(wave_obs2) * 1e4 / (1. + z))

            # Convert to photon energies since that what we work with internally
            E12 = h_p * c / (wave2[0] * 1e-8) / erg_per_ev
            E22 = h_p * c / (wave2[1] * 1e-8) / erg_per_ev

            # [enu] = erg/s/cm^3
            enu2 = self.get_emissivity(z, Emin=E22, Emax=E12)

        # Need angular diameter distance and H(z) for all that follows
        d = self.cosm.ComovingRadialDistance(0., z)           # [cm]
        Hofz = self.cosm.HubbleParameter(z)                   # [s^-1]

        ##
        # Must retrieve redshift-dependent k given fixed angular scale.
        if scale_units.lower() in ['arcsec', 'arcmin', 'deg']:
            rad = scale * (np.pi / 180.)

            # Convert to degrees retroactively
            if scale_units == 'arcsec':
                rad /= 3600.
            elif scale_units == 'arcmin':
                rad /= 60.
            elif scale_units.startswith('deg'):
                pass
            else:
                raise NotImplemented('Unrecognized scale_units={}'.format(
                    scale_units
                ))

            q = 2. * np.pi / rad
            k = q / (d / cm_per_mpc)
        elif scale_units.lower() in ['l', 'ell']:
            k = scale / (d / cm_per_mpc)
        else:
            raise NotImplemented('Unrecognized scale_units={}'.format(
                scale_units))

        ##
        # First: compute 3-D power spectrum
        if include_2h:
            ps3d = self.get_ps_2h(z, k, wave1=wave1, wave2=wave2, raw=False,
                nebular_only=False)
        else:
            ps3d = np.zeros_like(k)

        if include_shot:
            ps_shot = self.get_ps_shot(z, k, wave1=wave1, wave2=wave2,
                raw=self.pf['pop_1h_nebular_only'],
                nebular_only=False)
            ps3d += ps_shot

        if include_1h:
            ps_1h = self.get_ps_1h(z, k, wave1=wave1, wave2=wave2,
                raw=not self.pf['pop_1h_nebular_only'],
                nebular_only=self.pf['pop_1h_nebular_only'],
                prof=prof)
            ps3d += ps_1h

        # Compute "cross-terms" in 1-halo contribution from central--satellite
        # pairs.
        if include_1h and self.is_satellite_pop:
            assert self.pf['pop_centrals_id'] is not None, \
                "Must provide ID number of central population!"

        # The 3-d PS should have units of luminosity^2 * cMpc^-3.
        # Yes, that's cMpc^-3, a factor of volume^2 different than what
        # we're used to (e.g., matter power spectrum).

        # Must convert length units from cMpc (inherited from HMF)
        # to cgs.
        # Right now, ps3d \propto n^2 Plin(k)
        # [ps3d] = (erg/s)^2 (cMpc)^-6 right now
        # Hence the (ps3d / cm_per_mpc) factors below to get in cgs units.

        ##
        # Angular scales in arcsec, arcmin, or deg
        if scale_units.lower() in ['arcsec', 'arcmin', 'deg']:

            # e.g., Kashlinsky et al. 2018 Eq. 1, 3
            # Note: no emissivities here.
            dfdz = c * self.cosm.dtdz(z) / 4. / np.pi / (1. + z)
            delsq = (k / cm_per_mpc)**2 * (ps3d / cm_per_mpc**3) * Hofz \
                / 2. / np.pi / c

            if is_band_int:
                integrand = 2. * np.pi * dfdz**2 * delsq / q**2
            else:
                integrand = 2. * np.pi * dfdz**2 * delsq / q**2

        # Spherical harmonics
        elif scale_units.lower() in ['l', 'ell']:
            # Fernandez+ (2010) Eq. A9 or 37
            if is_band_int:
                # [ps3d] = cm^3
                integrand = c * (ps3d / cm_per_mpc**3) / Hofz / d**2 \
                    / (1. + z)**4 / (4. * np.pi)**2
            # Fernandez+ (2010) Eq. A10
            else:
                integrand = c * (ps3d / cm_per_mpc**3) / Hofz / d**2 \
                    / (1. + z)**2 / (4. * np.pi)**2
        else:
            raise NotImplemented('scale_units={} not implemented.'.format(scale_units))

        return integrand

    def _guess_Mmin(self):
        """
        Super non-general at the moment sorry.
        """

        fn = self.pf['feedback_LW_guesses']

        if fn is None:
            return None

        if type(fn) is str:
            anl = ModelSet(fn)
        elif isinstance(fn, ModelSet):
            anl = fn
        else:
            zarr, Mmin = fn

            if np.all(np.logical_or(np.isinf(Mmin), np.isnan(Mmin))):
                print("Provided Mmin guesses are all infinite or NaN.")
                return None

            return np.interp(self.halos.tab_z, zarr, Mmin)

        # HARD CODING FOR NOW
        blob_name = 'popIII_Mmin'
        Mmin = anl.ExtractData(blob_name)[blob_name]
        zarr = anl.get_ivars(blob_name)[0]

        ##
        # Remember: ModelSet will have {}'s and guesses_from will not.
        ##
        kw = {par: self.pf[par] \
            for par in self.pf['feedback_LW_guesses_from']}

        score = 0.0
        pid = self.pf['feedback_LW_sfrd_popid']
        for k, par in enumerate(self.pf['feedback_LW_guesses_from']):
            p_w_id = '%s{%i}' % (par, pid)

            if p_w_id not in anl.parameters:
                continue

            ind = list(anl.parameters).index(p_w_id)

            vals = anl.chain[:,ind]

            score += np.abs(np.log10(vals) - np.log10(kw[par]))

        best = np.argmin(score)

        return np.interp(self.halos.tab_z, zarr, Mmin[best])
