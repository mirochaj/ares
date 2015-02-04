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
    #f = open('%s/.ares/labels.py' % HOME, 'r')
    f, filename, data = imp.find_module('labels', ['%s/.ares/' % HOME])
    custom_labels = imp.load_module('labels.py', f, filename, data).pf
else:
    custom_labels = {}
    
#
## Common axis labels
label_flux_nrg = r'$J_{\nu} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'
label_flux_phot = r'$J_{\nu} \ (\mathrm{s}^{-1} \ \mathrm{cm}^{-2} \ \mathrm{Hz}^{-1} \ \mathrm{sr}^{-1})$'    
label_nrg = r'$h\nu \ (\mathrm{eV})$'
label_heat_mpc = r'$\epsilon_{\mathrm{heat}} \ (\mathrm{erg} \ \mathrm{s}^{-1} \ \mathrm{cMpc}^{-3})$'
label_dTb = r'$\delta T_b \ (\mathrm{mK})$'
label_dTbdnu = r'$d (\delta T_{\mathrm{b}}) / d\nu \ (\mathrm{mK/MHz})$'

labels = \
{
 'nu': r'$\nu \ (\mathrm{MHz})$',
 't_myr': r'$t \ (\mathrm{Myr})$',
 'flux': label_flux_phot, 
 'flux_E': label_flux_nrg, 
 'E': label_nrg,  
 'heat_mpc': label_heat_mpc,  
 'dTb': label_dTb,
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
 
 'spectrum_logN': r'$\log_{10} N_{\mathrm{H}}$',
 'spectrum_alpha': r'$\alpha$',
 'source_temperature': r'$T_{\ast}$',
 'z': r'$z$',
 'igm_heat_h_1': r'$\epsilon_{X, \mathrm{HI}}$',
 'igm_heat_he_1': r'$\epsilon_{X, \mathrm{HI}}$',
 'igm_heat_he_2': r'$\epsilon_{X, \mathrm{HI}}$', 
 'igm_heat': r'$\epsilon_X$', 
 'cgm_Gamma_h_1': r'$\Gamma_{\mathrm{HI},\mathrm{cgm}}}$',
 'igm_Gamma_h_1': r'$\Gamma_{\mathrm{HI},\mathrm{igm}}}$',
 'igm_Gamma_he_1': r'$\Gamma_{\mathrm{HeI}}$',
 'igm_Gamma_he_2': r'$\Gamma_{\mathrm{HeII}}$',

 'igm_gamma_h_1': r'$\gamma_{\mathrm{HI}}$',
 'igm_gamma_he_1': r'$\gamma_{\mathrm{HeI}}$',
 'igm_gamma_he_2': r'$\gamma_{\mathrm{HeII}}$',
 
 # Partial secondary ionizations
 'igm_gamma_h_1_h_1': r'$\gamma_{\mathrm{HI},\mathrm{HI}}$',
 'igm_gamma_h_1_he_1': r'$\gamma_{\mathrm{HI}, \mathrm{HeI}}$',
 'igm_gamma_h_1_he_2': r'$\gamma_{\mathrm{HI}, \mathrm{HeII}}$',
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
 'dTb': label_dTb,
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

labels.update(custom_labels)
