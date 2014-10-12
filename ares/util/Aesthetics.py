"""

Aesthetics.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 16:15:52 MDT 2014

Description: 

"""

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
 'flux': label_flux_phot, 
 'flux_E': label_flux_nrg, 
 'E': label_nrg,  
 'heat_mpc': label_heat_mpc,  
 'dTb': label_dTb,
 'dTbdnu': label_dTbdnu,
 'fX': r'$f_X$',
 'fstar': r'$f_{\ast}$',
 'Nion': r'$N_{\mathrm{ion}}$',
 'Tmin': r'$T_{\mathrm{min}} \ (\mathrm{K})$',
 'Nlw': r'$N_{\alpha}$',
 'z': r'$z$',
 'igm_heat': r'$\epsilon_X \ (\mathrm{erg} \ \mathrm{cm}^{-3} \ \mathrm{s}^{-1})$',
 'cgm_Gamma': r'$\Gamma_{\mathrm{HI}} \ (\mathrm{s}^{-1})$',
 'Tk': r'$T_K \ (\mathrm{K})$',
}    
##
#

history_elements = \
{
 'igm_h_1': r'$x_{\mathrm{HI}}',
 'igm_h_2': r'$x_{\mathrm{HII}}',
 'igm_he_1': r'$x_{\mathrm{HeI}}',
 'igm_he_2': r'$x_{\mathrm{HeII}}',
 'igm_he_3': r'$x_{\mathrm{HeIII}}',
 'igm_Tk': r'$T_K$',
 'igm_heat': r'$\epsilon_X$',
 'cgm_Gamma': r'$\Gamma_{\mathrm{HI}}$',
 'cgm_h_2': r'$x_{\mathrm{HII}}$',
 'Ts': r'$T_S$',
 'dTb': label_dTb,
 'z': r'$z$',
 'Ja': r'$J_{\alpha}$', 
 
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

labels.update(history_elements)
labels.update(tanh_parameters)
