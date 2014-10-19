"""

NormalizeSED.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Sep 15 12:37:21 MDT 2013

Description: 

"""

import rt1d
import numpy as np
from scipy.integrate import quad
from rt1d.physics.Constants import E_LyA, E_LL, erg_per_ev, g_per_msun, \
    s_per_yr

all_bands = ['lw', 'ion', 'xray']

def boxcar(x, x1, x2):
    if x <= x1:
        return False
    elif x >= x2: 
        return False
    else:
        return True
        
def emission_bands(Emin, Emax, Ex):
    """
    Determines emission bands for any (?) SED.
    """
    
    Emin = np.array(Emin)
    Emax = np.array(Emax)
        
    # Test energies for each band
    Ebands = np.array([np.mean([E_LyA, E_LL]), E_LL, np.mean([Ex, max(Emax)])])
            
    # Loop over (Emin, Emax) pairs, do they coincide with LW, ion, or Xray?
    tmp = []
    for i in range(len(Emin)):
        for j, nrg in enumerate(Ebands):
            if boxcar(nrg, Emin[i], Emax[i]):
                tmp.append(all_bands[j])
    
    bands = []
    for band in all_bands:
        if band in tmp:
            bands.append(band)
        
    return bands

def norm_sed(pop, grid):
    """
    Return normalizations for SED in LW, UV, and X-ray bands.
    
    Only works for single-component SEDs at the moment.
    
    Parameters
    ----------
    pop : *Population instance (Stellar, BlackHole, etc.)
    grid : rt1d.static.Grid instance
    
    Returns
    -------
    Dictionary containing normalization factors for all SED emission bands.
    
    """
    
    pf = pop.pf
    approx_src = pf['approx_lya'] and pf['approx_xray']
    
    cosm = pop.cosm
    b_per_g = 1. / cosm.g_per_baryon
    
    # Defaults (approximate source)
    Nlw = pf['Nlw']
    Elw = 0.5 * (E_LL + E_LyA)
    cLW, erg_per_LW = Nlw * Elw * b_per_g, Elw
    
    Nion = pf['Nion']
    Eion = pf['uv_Eavg']
    cUV, erg_per_UV = Nion * Eion * b_per_g, Eion
    
    Ex = pf['xray_Eavg']
    cX, erg_per_X = pf['cX'] * s_per_yr / g_per_msun, Ex * erg_per_ev
    Nx = cX / Ex / erg_per_ev / b_per_g
    
    # Return now if not treating the SED explicitly
    if approx_src:
        return {'rs':None, 
            'cLW':cLW, 'erg_per_LW':erg_per_LW, 'Nlw': Nlw, 'Elw': Elw,
            'cUV':cUV, 'erg_per_UV':erg_per_UV, 'Nion': Nion, 'Eion': Eion,
            'cX':cX,   'erg_per_X': erg_per_X, 'Nx': Nx, 'Ex': Ex}

    #
    ## REAL NORMALIZATION STARTS NOW
    #

    norm_by = pf['norm_by']
    
    # Read in some stuff for convenience
    alpha = pf['spectrum_alpha']
    Emin = pf['spectrum_Emin']
    Emax = pf['spectrum_Emax']
    EminNorm = pf['spectrum_EminNorm']
    EmaxNorm = pf['spectrum_EmaxNorm']
    
    if EminNorm is None:
        EminNorm = Emin
    if EmaxNorm is None:
        EmaxNorm = Emax
    
    # rt1d wants lists for spectrum_* parameters
    if type(alpha) is not list:
        alpha = list([alpha])
    if type(Emin) is not list:
        Emin = list([Emin])
    if type(Emax) is not list:
        Emax = list([Emax])
    if type(EminNorm) is not list:
        EminNorm = list([EminNorm])
    if type(EmaxNorm) is not list:
        EmaxNorm = list([EmaxNorm])
                              
    # rt1d tries to take care of normalization. Force it not to.
    tmp = pf.copy()
    tmp.update({'spectrum_EminNorm':None, 'spectrum_EmaxNorm':None})
    rs = rt1d.sources.RadiationSource(grid=grid, init_tabs=False, **tmp)
    
    # Number of spectral components
    Nc = rs.N
    
    if Nc > 1:
        print 'WARNING: SED normalization only has support for single-component SEDs.'
        return 
                
    # Determine which bands contain emission
    bands = emission_bands(Emin, Emax, pf['xray_Emin'])
                
    # If emission only in one band, normalize by that band's properties
    if len(bands) == 1:
        norm_by == bands[0]
        
    # Normalize by Nlw
    if norm_by == 'lw':
        
        # LW radiation
        Nlw = pf['Nlw']
        norm_num = Nlw / quad(lambda E: rs.Spectrum(E) / E, E_LyA, E_LL)[0]
        erg_per_LW = norm_num * quad(rs.Spectrum, E_LyA, E_LL)[0] \
            * erg_per_ev / Nlw
        cLW = Nlw * erg_per_LW * b_per_g
        Elw = erg_per_LW / erg_per_ev

        # Ionizing radiation
        if 'ion' in bands:
            Nion = norm_num * quad(lambda E: rs.Spectrum(E) / E, E_LL,
                pf['xray_Emin'])[0]
            erg_per_UV = norm_num * quad(rs.Spectrum, E_LL, 
                pf['xray_Emin'])[0] * erg_per_ev / Nion
            cUV = norm_num * erg_per_UV * b_per_g
            Eion = erg_per_UV / erg_per_ev
        elif pf['is_ion_src_cgm']:
            pass
        else:
            cUV = erg_per_UV = 0
                       
        # X-rays
        if 'xray' in bands:
            Nx = norm_num * quad(lambda E: rs.Spectrum(E) / E,
                pf['xray_Emin'], max(Emax))[0]
                    
            erg_per_X = norm_num * quad(rs.Spectrum, 
                pf['xray_Emin'], np.inf)[0] * erg_per_ev / Nx   
            cX = norm_num * erg_per_UV * b_per_g 
        elif pf['is_heat_src_igm']:
            pass    
        else:
            cX = erg_per_X = 0.0
         
    # Normalize by Nion        
    elif norm_by == 'ion':
        
        # Ionizing radiation
        Nion = pf['Nion']
        norm_num = Nion \
            / quad(lambda E: rs.Spectrum(E) / E, E_LL, pf['xray_Emin'])[0]
        erg_per_UV = norm_num * quad(rs.Spectrum, E_LL, pf['xray_Emin'])[0] \
            * erg_per_ev / Nion
        cUV = Nion * erg_per_UV * b_per_g
        Eion = erg_per_UV / erg_per_ev

        # Lyman-Werner
        if 'lw' in bands:
            Nlw = norm_num * quad(lambda E: rs.Spectrum(E) / E, E_LyA, E_LL)[0]
            erg_per_LW = norm_num * quad(rs.Spectrum, E_LyA, E_LL)[0] \
                * erg_per_ev / Nlw
            cLW = norm_num * erg_per_LW * b_per_g
            Elw = erg_per_LW / erg_per_ev
        elif pf['is_lya_src']:
            pass
        else:
            cLW = erg_per_LW = 0
                       
        # X-rays
        if 'xray' in bands:
            Nx = norm_num * quad(lambda E: rs.Spectrum(E) / E,
                pf['xray_Emin'], max(Emax))[0]
            erg_per_X = norm_num * quad(rs.Spectrum, 
                pf['xray_Emin'], np.inf)[0] * erg_per_ev / Nx
            cX = norm_num * erg_per_UV * b_per_g 
        elif pf['is_heat_src_igm']:
            pass
        else:
            cX = erg_per_X = 0.0
            
    # Normalize by cX    
    elif norm_by == 'xray':
        
        iX = 0
                
        # cX represents erg / g emitted in EminNorm -> EmaxNorm band
        if pf['model'] == -1:
            cX = pf['cX'] * s_per_yr / g_per_msun
            
            norm_nrg = cX / quad(rs.Spectrum, EminNorm[iX], EmaxNorm[iX])[0]

            Nx = norm_nrg * quad(lambda E: rs.Spectrum(E) / E, 
                 max(pf['xray_Emin'], Emin[iX]), Emax[iX])[0] \
                 / erg_per_ev / b_per_g    
            
            erg_per_X = norm_nrg * quad(rs.Spectrum,
                pf['xray_Emin'], Emax[iX])[0] / Nx / b_per_g
            
            Ex = erg_per_X / erg_per_ev
            
            norm_num = Nx / quad(lambda E: rs.Spectrum(E) / E, 
                EminNorm[iX], EmaxNorm[iX])[0]
            
            # Ionizing radiation
            if 'ion' in bands:
                Nion = norm_num * quad(lambda E: rs.Spectrum(E) / E, E_LL,
                    pf['xray_Emin'])[0]
                erg_per_UV = norm_num * quad(rs.Spectrum, E_LL, 
                    pf['xray_Emin'])[0] * erg_per_ev / Nion 
                cUV = norm_num * erg_per_UV * b_per_g
                Eion = erg_per_UV / erg_per_ev
            elif pf['is_ion_src_cgm']:
                pass
            else:
                cUV = erg_per_UV = 0
                
            # Lyman-Werner
            if 'lw' in bands:
                Nlw = norm_num * quad(lambda E: rs.Spectrum(E) / E, E_LyA, E_LL)[0]
                erg_per_LW = norm_num * quad(rs.Spectrum, E_LyA, E_LL)[0] \
                    * erg_per_ev / Nlw
                cLW = norm_num * erg_per_LW * b_per_g
                Elw = erg_per_LW / erg_per_ev
            elif pf['is_lya_src']:
                pass
            else:
                cLW = erg_per_LW = 0
        else:
            return {'rs':None, 
                'cLW':cLW, 'erg_per_LW':erg_per_LW, 'Nlw': Nlw, 'Elw': Elw,
                'cUV':cUV, 'erg_per_UV':erg_per_UV, 'Nion': Nion, 'Eion': Eion,
                'cX':cX,   'erg_per_X': erg_per_X, 'Nx': Nx, 'Ex': Ex}
            #raise NotImplemented('norm_by=xray and model != -1 not yet implemented.')
        
    else:
        print "WARNING: norm_by=%s is not a valid option!" % pf['norm_by']
        #raise ValueError('norm_by=%s is not a valid option!' % pf['norm_by'])
                
    return {'rs':rs, 
        'cLW':cLW, 'erg_per_LW':erg_per_LW, 'Nlw': Nlw, 'Elw': Elw,
        'cUV':cUV, 'erg_per_UV':erg_per_UV, 'Nion': Nion, 'Eion': Eion,
        'cX':cX,   'erg_per_X': erg_per_X, 'Nx': Nx, 'Ex': Ex}
                
    