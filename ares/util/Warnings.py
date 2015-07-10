"""

Warnings.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 13 14:31:51 MDT 2014

Description: 

"""

import sys
import numpy as np
from .PrintInfo import twidth, line, tabulate

separator = '|'*twidth
separator2 = '-'*twidth

def dt_error(grid, z, q, dqdt, new_dt, cell, method):
    
    print ""    
    print line(separator)
    print line('WARNING: something wrong with the time-step')    
    print line(separator)
    
    print line(separator2)    
    if new_dt <= 0:
        print line("current dt  : %.4e" % new_dt)
    else:
        print line("current dt  : NaN or inf")
                
    print line(separator2)
    print line("method      : %s" % method)
    print line("cell #      : %i" % cell)
    if z is not None:
        print line("redshift    : %.4g" % z)
    print line(separator2)  
     
    cols = ['value', 'derivative']
    
    rows = []
    data = []
    for i in range(len(grid.qmap)):
        name = grid.qmap[i]
        rows.append(name)                
        data.append([q[cell][i], dqdt[cell][i]])
            
    # Print quantities and their rates of change
    tabulate(data, rows, cols, cwidth=12)  

    print line(separator2)        

    print line(separator)
    print ""
    
    sys.exit(1)
    
    
tab_warning = \
"""
WARNING: must supply redshift_bins or tau_table to compute the X-ray background 
flux on-the-fly."""

wrong_tab_type = \
"""
WARNING: Supplied tau_table does not have logarithmically spaced redshift bins!
"""

hmf_no_tab = \
"""
No halo mass function table found. Run glorb/examples/generate_hmf_tables.py
to create a lookup table, then, either set an environment variable $ARES that 
points to your glorb install directory, or supply the path to the resulting
table by hand via the hmf_table parameter. You may also want to check out 
https://bitbucket.org/mirochaj/glorb/Downloads for standard HMF tables.
"""

def tau_tab_z_mismatch(igm, zmin_ok, zmax_ok, ztab):    
    print ""    
    print line(separator)
    print line('WARNING: optical depth table shape mismatch (in redshift)')    
    print line(separator)        
    
    if type(igm.tabname) is dict:
        which = 'dict'
    else:
        which = 'tab'
        print line("found       : %s" % igm.tabname[igm.tabname.rfind('/')+1:])
    
    print line("zmin (pf)   : %g" % igm.pf['final_redshift'])
    print line("zmin (%s)  : %g" % (which, ztab.min()))
    print line("zmax (pf)   : %g" % \
        (min(igm.pf['first_light_redshift'], igm.pf['initial_redshift'])))
    print line("zmax (%s)  : %g" % (which, ztab.max()))

    if not zmin_ok:
        print line("this is OK  : we'll transition to an on-the-fly tau calculator at z=%.2g" % ztab.min())
        if (0 < igm.pf['EoR_xavg'] < 1):
            print line("            : or whenever x > %.1e, whichever comes first" % igm.pf['EoR_xavg'])

    print line(separator)

def tau_tab_E_mismatch(igm, Emin_ok, Emax_ok, Etab):    
    print ""    
    print line(separator)
    print line('WARNING: optical depth table shape mismatch (in photon energy)')    
    print line(separator)        
    
    if type(igm.tabname) is dict:
        which = 'dict'
    else:
        which = 'tab'
        print line("found       : %s" % igm.tabname[igm.tabname.rfind('/')+1:])
    
    print line("Emin (pf)   : %g" % igm.pf['spectrum_Emin'])
    print line("Emin (%s)  : %g" % (which, Etab.min()))
    print line("Emax (pf)   : %g" % igm.pf['spectrum_Emax'])
    print line("Emax (%s)  : %g" % (which, Etab.max())) 

    if Etab.min() < igm.pf['spectrum_Emin']:
        print line("this is OK  : we'll discard E < %.2e eV entries in table" \
            % igm.pf['spectrum_Emin'])

    if Etab.max() > igm.pf['spectrum_Emax']:
        print line("this is OK  : we'll discard E > %.2e eV entries in table" \
            % igm.pf['spectrum_Emax']) 

    print line(separator)

def no_tau_table(igm):
    print ""    
    print line(separator)
    print line('ERROR (no optical depth table)')    
    print line(separator)        
    print line("looking for : %s" % igm.tau_name()[0])
    if igm.pf['tau_prefix'] is not None:
        print line("in          : %s" % igm.pf['tau_prefix'])
    elif glorb_dir is not None:
        print line("in          : %s/input/optical_depth" % glorb_dir)
    else:
        print line("in          : nowhere! set $ARES or tau_prefix")

    print line(separator)

def negative_SFRD(z, Tmin, fstar, dfcolldt, sfrd):
    print ""
    print line(separator)
    print line('ERROR (SFRD < 0)')    
    print line(separator)        
    print line("z           : %.3g" % z)
    print line("Tmin        : %.3e" % Tmin)
    print line("fstar       : %.3e" % fstar)
    print line("dfcoll / dt : %.3e" % dfcolldt)    
    print line("SFRD        : %.3e" % sfrd)    
    print line(separator)        

def tau_quad(igm):
    print ""
    print line(separator)
    print line('ERROR (SFRD < 0)')    
    print line(separator)        
    print line("z           : %.3g" % z)
    print line("Tmin        : %.3e" % Tmin)
    print line("fstar       : %.3e" % fstar)
    print line("dfcoll / dt : %.3e" % dfcolldt)    
    print line("SFRD        : %.3e" % sfrd)    
    print line(separator)


modelgrid_loadbalance = ""











    
        