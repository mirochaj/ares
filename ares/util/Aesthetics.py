"""

Aesthetics.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 16:15:52 MDT 2014

Description:

"""

import os, imp, re
import numpy as np
from matplotlib import cm
from .ParameterFile import par_info
from matplotlib.colors import ListedColormap

# Charlotte's color-maps
_charlotte1 = ['#301317','#3F2A3D','#2D4A60','#036B66','#48854D','#9D9436','#F69456']
_charlotte2 = ['#001316', '#2d2779', '#9c207e', '#c5492a', '#819c0c', '#3dd470', '#64cdf6']

cmap_charlotte1 = ListedColormap(_charlotte1, name='charlotte1')
cmap_charlotte2 = ListedColormap(_charlotte2, name='charlotte2')

_zall = np.arange(4, 11, 1)

_znormed = (_zall - _zall[0]) / float(_zall[-1] - _zall[0])
_ch_c1 = cm.get_cmap(cmap_charlotte1, _zall.size)
_ch_c2 = cm.get_cmap(cmap_charlotte2, _zall.size)
_normz = lambda zz: (zz - _zall[0]) / float(_zall[-1] - _zall[0])

colors_charlotte1 = lambda z: _ch_c1(_normz(z))
colors_charlotte2 = lambda z: _ch_c2(_normz(z))


# Load custom defaults
HOME = os.environ.get('HOME')
if os.path.exists('{!s}/.ares/labels.py'.format(HOME)):
    f, filename, data = imp.find_module('labels', ['{!s}/.ares/'.format(HOME)])
    custom_labels = imp.load_module('labels.py', f, filename, data).pf
else:
    custom_labels = {}

prefixes = ['igm_', 'cgm_']

#
## Common axis labels
label_flux_nrg = r'$J_{\nu} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'
label_flux_phot = r'$J_{\nu} \ (\mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'
label_flux_nw = r'$J_{\nu} \ [\mathrm{nW} \ \mathrm{m}^{-2} \ \mathrm{sr}^{-1}]$'
label_logflux_nw = r'$\log_{10} (J_{\nu} / [\mathrm{nW} \ \mathrm{m}^{-2} \ \mathrm{sr}^{-1}])$'
label_power_nw = r'$q^2 P(q)/(2\pi) \ (\mathrm{nW}^2 \ \mathrm{m}^{-4} \ \mathrm{sr}^{-2})$'
label_power_nw_sqrt = r'$\sqrt{q^2 P(q)/(2\pi)} \ (\mathrm{nW} \ \mathrm{m}^{-2} \ \mathrm{sr}^{-1})$'
label_power_Cl_sqrt = r'$\left[ l(l+1) C_l^{\nu \nu^{\prime}} / (2\pi) \right]^{1/2} \ (\mathrm{nW} \ \mathrm{m}^{-2} \ \mathrm{sr}^{-1})$'
label_power_Cl = r'$l(l+1) C_l^{\nu \nu^{\prime}} / (2\pi) \ (\mathrm{nW}^{2} \ \mathrm{m}^{-4} \ \mathrm{sr}^{-2})$'
label_flux_nuInu = r'$\nu I_{\nu} \ (\mathrm{nW} \ \mathrm{m}^{-2} \ \mathrm{sr}^{-1})$'
label_nrg = r'$h\nu \ (\mathrm{eV})$'
label_heat_mpc = r'$\epsilon_{\mathrm{heat}} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3})$'
label_dTbdnu = r'$d (\delta T_{\mathrm{b}}) / d\nu \ (\mathrm{mK/MHz})$'
label_MAR = r'$\dot{M}_h \ [M_{\odot} \ \mathrm{yr}^{-1}]$'
label_logMAR = r'$\log_{10} \left(\dot{M}_h / [M_{\odot} \ \mathrm{yr}^{-1}]\right)$'
label_L_nu = r'$L_{\nu} \ [\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1}]$'
label_L_lam = r'$L_{\lambda} \ [\mathrm{erg} \ \mathrm{s}^{-1} \ \AA^{-1}]$'

states = \
{
'h_1': r'$x_{\mathrm{HI}}$',
'h_2': r'$x_{\mathrm{HII}}$',
'he_1': r'$x_{\mathrm{HeI}}$',
'he_2': r'$x_{\mathrm{HeII}}$',
'he_3': r'$x_{\mathrm{HeIII}}$',
'Tk': r'$T_K$',
}

rates = \
{
 'k_ion': r'$\kappa_{\mathrm{ion}}$',
 'k_ion2': r'$\kappa_{\mathrm{ion, sec}}$',
 'k_heat': r'$\kappa_{\mathrm{heat}}$',
 'k_diss': r'$\kappa_{\mathrm{diss}}$',
}

derived = \
{
 'Ts': r'$T_S$',
 'dTb': r'$\delta T_b \ (\mathrm{mK})$',
 #'hwhm_diff': r'$\Delta \nu_{\min}$',
 #'squash': r'$\delta T_b(\nu_{\min}) / \mathrm{FWHM}$',
 'hwhm_diff': r'$\mathcal{A} \ (\mathrm{MHz})$',
 'squash': r'$\mathcal{W} \ (\mathrm{mK} \ \mathrm{MHz}^{-1})$',
 'fwhm': r'$\mathrm{FWHM}$',
 'fwqm': r'$\mathrm{FWQM}$',
 'mean_slope': r'$\langle \delta T_b^{\prime} \rangle$',
 'mean_slope_hi': r'$\langle \delta T_b^{\prime} \rangle_{\mathrm{hi}}$',
 'mean_slope_lo': r'$\langle \delta T_b^{\prime} \rangle_{\mathrm{lo}}$',
}

labels = {}
labels.update(states)
labels.update(rates)
labels.update(derived)

# Also account for prefixes
labels_w_prefix = {}
for prefix in prefixes:
    for key in labels:
        labels_w_prefix['{0!s}{1!s}'.format(prefix, key)] = labels[key]

labels.update(labels_w_prefix)

common = \
{
 'nu': r'$\nu \ (\mathrm{MHz})$',
 't_myr': r'$t \ (\mathrm{Myr})$',
 'flux': label_flux_phot,
 'flux_E': label_flux_nrg,
 'flux_nW': label_flux_nw,
 'logflux_nW': label_logflux_nw,
 'power_nirb': label_power_nw,
 'power_nirb_sqrt': label_power_nw_sqrt,
 'power_nirb_Cl_sqrt': label_power_Cl_sqrt,
 'power_nirb_Cl': label_power_Cl,
 'angular_scale_q_min': r'$2 \pi / q \ [\mathrm{arcmin}]$',
 'angular_scale_q_sec': r'$2 \pi / q \ [\mathrm{arcsec}]$',
 'angular_scale_l': r'Multipole moment, $l$',
 'flux_nuInu': label_flux_nuInu,
 'intensity_AA': r'$\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{\AA}^{-1}$',
 'lambda_AA': r'$\lambda \ (\AA)$',
 'L_nu': label_L_nu,
 'L_lam': label_L_lam,
 'E': label_nrg,
 'heat_mpc': label_heat_mpc,
 'dTbdnu': label_dTbdnu,
 'fX': r'$f_X$',
 'fstar': r'$f_{\ast}$',
 'fesc': r'$f_{\mathrm{esc}}$',
 'Nion': r'$N_{\mathrm{ion}}$',
 'Tmin': r'$T_{\mathrm{min}}$',
 'MAR': label_MAR,
 'logMAR': label_logMAR,

 'Nlw': r'$N_{\alpha}$',
 'fbh': r'$f_{\bullet}$',
 'xi_XR': r'$\xi_{X}$',
 'xi_LW': r'$\xi_{\mathrm{LW}}$',
 'xi_UV': r'$\xi_{\mathrm{ion}}$',
 'sfrd': r'$\dot{\rho}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}]$',
 'sfr': r'$\dot{M}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1}]$',
 'logsfr': r'$\log_{10} \dot{M}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1}]$',
 'emissivity': r'$\epsilon \ [\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3}]$',
 'nh': r'$n_h \ [\mathrm{cMpc}^{-3}]$',
 'extinction_redshift': r'$z_{\mathrm{ext}}$',

 'source_logN': r'$\log_{10} N_{\mathrm{H}}$',
 'source_alpha': r'$\alpha$',
 'source_temperature': r'$T_{\ast}$',
 'z': r'$z$',
 'igm_k_heat_h_1': r'$\epsilon_{X, \mathrm{HI}}$',
 'igm_k_heat_he_1': r'$\epsilon_{X, \mathrm{HI}}$',
 'igm_k_heat_he_2': r'$\epsilon_{X, \mathrm{HI}}$',
 'igm_k_heat': r'$\epsilon_X$',
 'cgm_k_ion_h_1': r'$\Gamma_{\mathrm{HI},\mathrm{cgm}}}$',
 'igm_k_ion_h_1': r'$\Gamma_{\mathrm{HI},\mathrm{igm}}}$',
 'igm_k_ion_he_1': r'$\Gamma_{\mathrm{HeI}}$',
 'igm_k_ion_he_2': r'$\Gamma_{\mathrm{HeII}}$',

 'igm_k_ion2_h_1': r'$\gamma_{\mathrm{HI}}$',
 'igm_k_ion2_he_1': r'$\gamma_{\mathrm{HeI}}$',
 'igm_k_ion2_he_2': r'$\gamma_{\mathrm{HeII}}$',

 # Partial secondary ionizations
 'igm_k_ion2_h_1_h_1': r'$\gamma_{\mathrm{HI},\mathrm{HI}}$',
 'igm_k_ion2_h_1_he_1': r'$\gamma_{\mathrm{HI}, \mathrm{HeI}}$',
 'igm_k_ion2_h_1_he_2': r'$\gamma_{\mathrm{HI}, \mathrm{HeII}}$',
 'Tk': r'$T_K \ (\mathrm{K})$',

 'tau_e': r'$\tau_e$',
 'tau_tot': r'$\tau_e$',
 'z_dec': r'$z_{\mathrm{dec}}$',

 'skewness_absorption': r'$\mu_{3, \mathrm{abs}}$',
 'kurtosis_absorption': r'$\mu_{4, \mathrm{abs}}$',
 'skewness_emission': r'$\mu_{3, \mathrm{em}}$',
 'kurtosis_emission': r'$\mu_{4, \mathrm{em}}$',

 'igm_initial_temperature': r'$T_0$',
}
##
#


history_elements = \
{
 'igm_h_1': r'$x_{\mathrm{HI}}$',
 'igm_h_2': r'$x_{\mathrm{HII}}$',
 'igm_he_1': r'$x_{\mathrm{HeI}}$',
 'igm_he_2': r'$x_{\mathrm{HeII}}$',
 'igm_he_3': r'$x_{\mathrm{HeIII}}$',
 'igm_Tk': r'$T_K$',
 'cgm_h_2': r'$Q_{\mathrm{HII}}$',
 'xavg': r'$\overline{x}_i$',
 'Ts': r'$T_S$',
 'z': r'$z$',
 'nu': r'$\nu$',
 'Ja': r'$J_{\alpha}$',
 'Jlw': r'$J_{\mathrm{LW}}$',
 'dTb': r'$\delta T_b \ (\mathrm{mK})$',
 'dlogTk_dlogt': r'$d\log T_K / d\log t$',

 'slope': r'$\delta^{\prime} T_b \ [\mathrm{mK} \ \mathrm{MHz}^{-1}]$',
 'curvature': r'$\delta^{\prime \prime} T_b \ [\mathrm{mK}^2 \ \mathrm{MHz}^{-2}]$',
}

tp_parameters = {}
hist_plus_derived = history_elements
hist_plus_derived.update(derived)
for key in hist_plus_derived:
    for tp in ['A', 'B', 'C', 'D', 'ZC']:
        if key in ['z', 'nu']:
            tp_parameters['{0!s}_{1!s}'.format(key, tp)] = \
                r'{0!s}_{{\mathrm{{{1!s}}}}}$'.format(hist_plus_derived[key][0:-1], tp)
        else:
            tp_parameters['{0!s}_{1!s}'.format(key, tp)] = \
                r'{0!s}(\nu_{{\mathrm{{{1!s}}}}})$'.format(hist_plus_derived[key][0:-1], tp)

for key in hist_plus_derived:
    for tp in ['A', 'B', 'C', 'D']:
        if key in ['z', 'nu']:
            tp_parameters['{0!s}_{1!s}p'.format(key, tp)] = \
                r'{0!s}_{{\mathrm{{{1!s}}}}}^{{\prime}}$'.format(hist_plus_derived[key][0:-1], tp)
        else:
            tp_parameters['{0!s}_{1!s}p'.format(key, tp)] = \
                r'{0!s}(\nu_{{\mathrm{{{1!s}}}}}^{{\prime}})$'.format(hist_plus_derived[key][0:-1], tp)


tanh_parameters = \
{
 'tanh_J0': r'$\left(J_0 / J_{21}\right)$',
 'tanh_Jz0': r'$z_J$',
 'tanh_Jdz': r'$\Delta z_J$',
 'tanh_T0': r'$T_0$',
 'tanh_Tz0': r'$z_T$',
 'tanh_Tdz': r'$\Delta z_T$',
 'tanh_x0': r'$\overline{x}_{i,0}$',
 'tanh_xz0': r'$z_x$',
 'tanh_xdz': r'$\Delta z_x$',
 'tanh_bias_freq': r'$b_{\nu}$',
 'tanh_bias_temp': r'$b_{\mathrm{T}}$',
}

gauss_parameters = \
{
 'gaussian_A': r'$A_0 \ (\mathrm{mK})$',
 'gaussian_nu': r'$\nu_0 \ (\mathrm{MHz})$',
 'gaussian_sigma': r'$\sigma_0 \ (\mathrm{MHz})$',
}

lf_parameters = \
{
 'MUV': r'$M_{\mathrm{UV}}$',
 'pop_lf_Mstar': r'$M_{\ast}$',
 'pop_lf_pstar': r'$\phi_{\ast}$',
 'pop_lf_alpha': r'$\alpha$',
 'Mpeak': r'$M_{\mathrm{peak}}$',
 'fpeak': r'$f_{\ast} (M_{\mathrm{peak}})$',
 'gamma': r'$\gamma$',
 'Mh': r'$M_h / M_{\odot}$',
 'Lh': r'$L_h / (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{Hz}^{-1})$',
}

pop_parameters = \
{
 'pop_Z': r'$Z/Z_{\odot}$',
 'pop_sfr': r'$\dot{M}_{\ast}$',
 'pop_lf_beta': r'$\Beta_{\mathrm{UV}}$',
 'pop_fstar': r'$f_{\ast}$',
 'pop_fobsc': r'$f_{\mathrm{obsc}}$',
 'fobsc': r'$f_{\mathrm{obsc}}$',
 'pop_acc_frac_stellar': r'$f_{\ast}^{\mathrm{acc}}$',
 'pop_acc_frac_metals': r'$f_Z^{\mathrm{acc}}$',
 'pop_acc_frac_gas': r'$f_g^{\mathrm{acc}}$',
 'pop_metal_retention': r'$f_{\mathrm{ret,Z}}$',
 'pop_abun_limit': r'$\mathcal{Z}_c$',
 'pop_bind_limit': r'$\mathcal{E}_c$',
 'pop_time_limit': r'$\mathcal{T}_c$',
}

sfe_parameters = \
{
 "lf": r'$\phi(M_{1600}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf": r'$\phi(M_{1600}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf_muv": r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf_mag": r'$\phi(M) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf_1500": r'$\phi(M_{1500}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf_1600": r'$\phi(M_{1600}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_smf": r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$',
}

for i in range(6):
    sfe_parameters['pq_func_par{}'.format(i)] = r'$p_{%i}$' % i

powspec = \
{
 'k': r'$k \ [\mathrm{cMpc}^{-1}]$',
 'dpow': r'$\overline{\delta T_b}^2 \Delta_{21}^2 \ \left[\mathrm{mK}^2 \right]$',
 'pow': r'$P(k)$',
}
other = \
{
 'load': 'processor #',
 'contrast': r'$1 - T_{\gamma} / T_S$',
}

labels.update(history_elements)
labels.update(tanh_parameters)
labels.update(gauss_parameters)
labels.update(other)
labels.update(common)
labels.update(lf_parameters)
labels.update(pop_parameters)
labels.update(tp_parameters)
labels.update(sfe_parameters)
labels.update(powspec)

# Add custom labels
labels.update(custom_labels)

def logify_str(s, sup=None):
    s_no_dollar = str(s.replace('$', ''))

    new_s = s_no_dollar

    if sup is not None:
        new_s += '[{!s}]'.format(sup_scriptify_str(s))

    return r'$\mathrm{log}_{10}' + new_s + '$'

def undo_mathify(s):
    return str(s.replace('$', ''))

def mathify_str(s):
    return r'${!s}$'.format(s)

class Labeler(object): # pragma: no cover
    def __init__(self, pars, is_log=False, extra_labels={}, **kwargs):
        self.pars = self.parameters = pars
        self.base_kwargs = kwargs
        self.extras = extra_labels

        self.labels = labels.copy()
        self.labels.update(self.extras)

        if type(is_log) == bool:
            self.is_log = {par:is_log for par in pars}
        else:
            self.is_log = {}
            for par in pars:
                if par in self.parameters:
                    k = self.parameters.index(par)
                    self.is_log[par] = is_log[k]
                else:
                    # Blobs are never log10-ified before storing to disk
                    self.is_log[par] = False

    def units(self, prefix):
        units = None
        for kwarg in self.base_kwargs:
            if not re.search(prefix, kwarg):
                continue

            if re.search('units', kwarg):
                units = self.base_kwargs[kwarg]

        return units

    def _find_par(self, popid, phpid):
        kwarg = None
        look_for_1 = '{{{}}}'.format(popid)
        look_for_2 = '[{}]'.format(phpid)
        for kwarg in self.base_kwargs:
            if phpid is not None:
                if self.base_kwargs[kwarg] == 'pq[{}]'.format(phpid):
                    break

        return kwarg.replace('{{{}}}'.format(popid), '')

    def label(self, par, take_log=False, un_log=False):
        """
        Create a pretty label for this parameter (if possible).
        """

        if par in self.labels:
            label = self.labels[par]

            if par in self.parameters:
                if take_log:
                    return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
                elif self.is_log[par] and (not un_log):
                    return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
                else:
                    return label
            else:
                return label

        prefix, popid, phpid = par_info(par)

        _par = par
        # Correct prefix is phpid is not None
        if phpid is not None:
            s = 'pq[{}]'.format(phpid)

            for _par in self.base_kwargs:
                if self.base_kwargs[_par] != s:
                    continue
                break

            prefix = _par

        units = self.units(prefix)

        label = None

        # Simplest case. Not popid, not a PQ, label found.
        if popid == phpid == None and (prefix in self.labels):
            label = self.labels[prefix]
        # Has pop ID number but is not a PQ, label found.
        elif (popid is not None) and (phpid is None) and (prefix in self.labels):
            label = self.labels[prefix]
        elif (popid is not None) and (phpid is None) and (prefix.strip('pop_') in self.labels):
            label = self.labels[prefix.strip('pop_')]
        # Has Pop ID, not a PQ, no label found.
        elif (popid is not None) and (phpid is None) and (prefix not in self.labels):
            try:
                hard = self._find_par(popid, phpid)
            except:
                hard = None

            if hard is not None:
                # If all else fails, just typset the parameter decently
                label = prefix
                #parnum = int(re.findall(r'\d+', prefix)[0]) # there can only be one
                #label = r'${0!s}\{{{1}\}}[{2}]<{3}>$'.format(hard.replace('_', '\_'),
                #    popid, phpid, parnum)
        # Is PQ, label found. Just need to parse []s.
        elif phpid is not None and (prefix in self.labels):
            parnum = list(map(int, re.findall(r'\d+', par.replace('[{}]'.format(phpid),''))))
            if len(parnum) == 1:
                label = r'${0!s}^{{\mathrm{{par}}\ {1}}}$'.format(\
                    undo_mathify(self.labels[prefix]), parnum[0])
            else:
                label = r'${0!s}^{{\mathrm{{par}}\ {1},{2}}}$'.format(\
                    undo_mathify(self.labels[prefix]), parnum[0], parnum[1])
        # Otherwise, just use number. Not worth the trouble right now.
        elif (popid is None) and (phpid is not None) and par.startswith('pq_'):
            label = 'par {}'.format(self.parameters.index(par))

        # Troubleshoot if label not found
        if label is None:
            label = prefix
            if re.search('pop_', prefix):
                if prefix[4:] in self.labels:
                    label = self.labels[prefix[4:]]
            else:
                label = r'${!s}$'.format(par.replace('_', '\_'))

        if par in self.parameters:
            #print('{0} {1} {2} {3}'.format(par, take_log, self.is_log[par],\
            #    un_log))
            if take_log:
                return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
            elif self.is_log[par] and (not un_log):
                return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
            else:
                return label

        return label
