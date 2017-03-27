"""

Aesthetics.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 16:15:52 MDT 2014

Description: 

"""

import os, imp, re
from .ParameterFile import par_info

# Load custom defaults    
HOME = os.environ.get('HOME')
if os.path.exists('%s/.ares/labels.py' % HOME):
    f, filename, data = imp.find_module('labels', ['%s/.ares/' % HOME])
    custom_labels = imp.load_module('labels.py', f, filename, data).pf
else:
    custom_labels = {}
    
prefixes = ['igm_', 'cgm_']    
    
#
## Common axis labels
label_flux_nrg = r'$J_{\nu} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'
label_flux_phot = r'$J_{\nu} \ (\mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'    
label_nrg = r'$h\nu \ (\mathrm{eV})$'
label_heat_mpc = r'$\epsilon_{\mathrm{heat}} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3})$'
label_dTbdnu = r'$d (\delta T_{\mathrm{b}}) / d\nu \ (\mathrm{mK/MHz})$'

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
 'hwhm_diff': r'$\Delta \mathrm{HWHM}$',
 'hwqm_diff': r'$\Delta \mathrm{HWQM}$',
 'fwhm': r'$\mathrm{FWHM}$',
 'fwqm': r'$\mathrm{FWQM}$',
}

labels = {}
labels.update(states)
labels.update(rates)
labels.update(derived)

# Also account for prefixes
labels_w_prefix = {}
for prefix in prefixes:
    for key in labels:
        labels_w_prefix['%s%s' % (prefix, key)] = labels[key]

labels.update(labels_w_prefix)

common = \
{
 'nu': r'$\nu \ (\mathrm{MHz})$',
 't_myr': r'$t \ (\mathrm{Myr})$',
 'flux': label_flux_phot, 
 'flux_E': label_flux_nrg, 
 'intensity_AA': r'$\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{\AA}^{-1}$', 
 'lambda_AA': r'$\lambda \ (\AA)$', 
 'E': label_nrg,  
 'heat_mpc': label_heat_mpc,  
 'dTbdnu': label_dTbdnu,
 'fX': r'$f_X$',
 'fstar': r'$f_{\ast}$',
 'fesc': r'$f_{\mathrm{esc}}$',
 'Nion': r'$N_{\mathrm{ion}}$',
 'Tmin': r'$T_{\mathrm{min}}$',
 'Nlw': r'$N_{\alpha}$',
 'fbh': r'$f_{\bullet}$',
 'xi_XR': r'$\xi_{X}$',
 'xi_LW': r'$\xi_{\mathrm{LW}}$',
 'xi_UV': r'$\xi_{\mathrm{ion}}$',
 'sfrd': r'$\dot{\rho}_{\ast} \ [M_{\odot} \ \mathrm{yr}^{-1} \ \mathrm{cMpc}^{-3}]$',
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
 
 'skewness_absorption': r'$\mu_{3, \mathrm{abs}}$',
 'kurtosis_absorption': r'$\mu_{4, \mathrm{abs}}$',
 'skewness_emission': r'$\mu_{3, \mathrm{em}}$',
 'kurtosis_emission': r'$\mu_{4, \mathrm{em}}$',
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
 
 'slope': r'$\delta^{\prime} T_b \ [\mathrm{mK} \ \mathrm{MHz}^{-1}]$',
 'curvature': r'$\delta^{\prime \prime} T_b \ [\mathrm{mK}^2 \ \mathrm{MHz}^{-2}]$', 
}

tp_parameters = {}
hist_plus_derived = history_elements
hist_plus_derived.update(derived)
for key in hist_plus_derived:
    for tp in ['B', 'C', 'D', 'ZC']:
        if key in ['z', 'nu']:
            tp_parameters['%s_%s' % (key, tp)] = \
                r'%s_{\mathrm{%s}}$' % (hist_plus_derived[key][0:-1], tp)
        else:
            tp_parameters['%s_%s' % (key, tp)] = \
                r'%s(\nu_{\mathrm{%s}})$' % (hist_plus_derived[key][0:-1], tp)

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
 'pop_acc_frac_stellar': r'$f_{\ast}^{\mathrm{acc}}$',
 'pop_acc_frac_metals': r'$f_Z^{\mathrm{acc}}$',
 'pop_acc_frac_gas': r'$f_g^{\mathrm{acc}}$',
 'pop_metal_retention': r'$f_{\mathrm{ret,Z}}$',
}

sfe_parameters = \
{
 "lf": r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "galaxy_lf": r'$\phi(M_{\mathrm{UV}}) \ [\mathrm{mag}^{-1} \ \mathrm{cMpc}^{-3}]$',
 "smf": r'$\phi(M_{\ast}) \ [\mathrm{dex}^{-1} \ \mathrm{cMpc}^{-3}]$',
}

for i in range(6):
    sfe_parameters['php_Mfun_par%i' % i] = r'$p_{%i}$' %i
    for j in range(6):
        sfe_parameters['php_Mfun_par%i_par%i' % (i,j)] = r'$p_{%i,%i}$' % (i,j)
        

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

# Add custom labels
labels.update(custom_labels)

def logify_str(s, sup=None):
    s_no_dollar = str(s.replace('$', ''))
    
    new_s = s_no_dollar
    
    if sup is not None:
        new_s += '[%s]' % sup_scriptify_str(s)
        
    return r'$\mathrm{log}_{10}' + new_s + '$'
    
def undo_mathify(s):
    return str(s.replace('$', ''))
    
def mathify_str(s):
    return r'$%s$' % s    
            
class Labeler(object):
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
        look_for_1 = '{%i}' % popid
        look_for_2 = '[%i]' % phpid
        for kwarg in self.base_kwargs:
            if phpid is not None:
                if self.base_kwargs[kwarg] == 'php[%i]' % phpid:
                    break
                
        return kwarg.replace('{%i}' % popid, '')
                
    def label(self, par, take_log=False, un_log=False):
        """
        Create a pretty label for this parameter (if possible).
        """
        
        if par in self.labels:
            return self.labels[par]
        
        prefix, popid, phpid = par_info(par)
                
        # Correct prefix is phpid is not None
        if phpid is not None:
            s = 'pq[%i]' % phpid
                
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
        # Has Pop ID, not a PQ, no label found.      
        elif (popid is not None) and (phpid is None) and (prefix not in self.labels):
            try:
                hard = self._find_par(popid, phpid)
            except:
                hard = None
        
            if hard is not None:    
                # If all else fails, just typset the parameter decently
                parnum = int(re.findall(r'\d+', prefix)[0]) # there can only be one
                label = r'$%s\{%i\}[%i]<%i>$' % (hard.replace('_', '\_'),
                    popid, phpid, parnum)    
        # Is PQ, label found. Just need to parse []s.
        elif phpid is not None and (prefix in self.labels):
            parnum = map(int, re.findall(r'\d+', par.replace('[%i]' % phpid,'')))
            if len(parnum) == 1:
                label = r'$%s^{\mathrm{par}\ %i}$' % \
                    (undo_mathify(self.labels[prefix]), parnum[0])
            else:
                label = r'$%s^{\mathrm{par}\ %i,%i}$' \
                    % (undo_mathify(self.labels[prefix]), parnum[0], parnum[1])
        # Otherwise, just use number. Not worth the trouble right now.
        elif (popid is None) and (phpid is not None) and par.startswith('pq_'):
            label = 'par %i' % (self.parameters.index(par))
            
        # Troubleshoot if label not found
        if label is None:         
            label = prefix
            if re.search('pop_', prefix):
                if prefix[4:] in self.labels:
                    label = self.labels[prefix[4:]]
            else:
                label = r'$%s$' % (par.replace('_', '\_'))
        
        if par in self.parameters:                    
            if take_log:        
                return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
            elif self.is_log[par] and (not un_log):
                return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
            else:
                return label    
        
        return label
        
        
    
