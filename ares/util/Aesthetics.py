"""

Aesthetics.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 16:15:52 MDT 2014

Description: 

"""

import os, imp

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
}

other = \
{
 'load': 'processor #',
 'contrast': r'$1 - T_{\gamma} / T_S$',
}

labels.update(history_elements)
labels.update(tanh_parameters)
labels.update(other)
labels.update(common)

# Add custom labels
labels.update(custom_labels)
