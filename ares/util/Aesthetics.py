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
 'curvature': r'$\delta^{\prime \prime} T_b \ [\mathrm{mK}^2 \ \mathrm{MHz}^{-2}]$',
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
 'Ja': r'$J_{\alpha}$', 
 'Jlw': r'$J_{\mathrm{LW}}$', 
}

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
}

pop_parameters = \
{
 'pop_Z': r'$Z/Z_{\odot}$',
 'pop_Tmin': r'$T_{\mathrm{min}}$',
 'pop_lf_beta': r'$\Beta_{\mathrm{UV}}$',
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
    
def make_label(name, take_log=False, labels=None):
    """
    Take a string and make it a nice (LaTeX compatible) axis label. 
    """
    
    if labels is None:
        labels = default_labels
        
    # Check to see if it has a population ID # tagged on the end
    # OR a redshift
    
    try:
        m = None
        prefix, z = param_redshift(name)
        prefix, popid = pop_id_num(prefix)
    except:
        m = re.search(r"\{([0-9])\}", name)
        z = popid = None
        
    if m is None:
        num = None
        prefix = name
        if prefix in labels:
            label = labels[prefix]
        else:
            label = r'%s' % prefix
    else:
        num = int(m.group(1))
        prefix = name.split(m.group(0))[0]
        
        if prefix in labels:
            label = r'$%s[z=%.2g]$' % (undo_mathify(labels[prefix]), z)
        else:
            label = r'%s' % prefix
        
    if take_log:        
        return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
    else:
        return label
        
def err_str(label, mu, err, log, labels=None):
    s = undo_mathify(make_label(label, log, labels))

    s += '=%.3g^{+%.2g}_{-%.2g}' % (mu, err[1], err[0])
    
    return r'$%s$' % s

class Labeler(object):
    def __init__(self, pars, is_log=False, **kwargs):
        self.pars = pars
        self.base_kwargs = kwargs
        
        if type(is_log) == bool:
            self.is_log = [is_log] * len(pars)
        else:
            self.is_log = is_log
        
    def units(self, prefix):
        units = None
        for kwarg in self.base_kwargs:
            if not re.search(prefix, kwarg):
                continue
            
            if re.search('units', kwarg):
                units = self.base_kwargs[kwarg]
        
        return units
                
    def label(self, par, take_log=False, un_log=False):
        """
        Create a pretty label for this parameter.
        """
        
        prefix, popid, redshift = par_info(par)
        
        units = self.units(prefix)
        
        label = None
                
        # Simplest case
        if popid == redshift == None and (par in labels):
            label = labels[prefix]
        # Has pop ID number
        elif (popid is not None) and (redshift is None) and (prefix in labels):
            label = labels[prefix]
        elif redshift is not None and (prefix in labels):
            label = r'$%s[z=%.2g]$' % (undo_mathify(labels[prefix]), redshift)
        
        # Troubleshoot if label not found
        if label is None:
            if re.search('pop_', prefix):
                if prefix[4:] in labels:
                    label = labels[prefix[4:]]
                else:
                    label = prefix
            else:
                label = prefix
            
        if take_log:        
            return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
        elif self.is_log[par] and (not un_log):
            return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
        else:
            return label    
        
        return label
        
        
    
