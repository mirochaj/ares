"""

Warnings.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 13 14:31:51 MDT 2014

Description: 

"""

import sys, os
import numpy as np
import sys, textwrap, os
from .PrintInfo import twidth, line, tabulate

try:
    from hmf import MassFunction
    have_hmf = True
except ImportError:
    have_hmf = False
    
try:
    import pycamb
    have_pycamb = True
except ImportError:
    have_pycamb = False
    
ARES = os.getenv('ARES')
have_ARES_env = ARES is not None

separator = '|'*twidth
separator2 = '-'*twidth

dt_msg = 'WARNING: something wrong with the time-step.'
gen_msg = 'WARNING: something wrong with solver.'
    
def dt_error(grid, z, q, dqdt, new_dt, cell, method, msg=dt_msg):
    
    print("")
    print(line(separator))
    print(line(msg))
    print(line(separator))
    
    print(line(separator2))
    if new_dt <= 0:
        print(line("current dt  : {0:.4e}".format(new_dt)))
    else:
        print(line("current dt  : NaN or inf"))
                
    print(line(separator2))
    print(line("method      : {!s}".format(method)))
    print(line("cell #      : {}".format(cell)))
    if z is not None:
        print(line("redshift    : {0:.4g}".format(z)))

    print(line(separator2))
     
    cols = ['value', 'derivative']
    
    rows = []
    data = []
    for i in range(len(grid.qmap)):
        name = grid.qmap[i]
        rows.append(name)                
        data.append([q[cell][i], dqdt[cell][i]])
            
    # Print quantities and their rates of change
    tabulate(data, rows, cols, cwidth=12)  

    print(line(separator2))

    print(line(separator))
    print("")
    
    sys.exit(1)
    
def solver_error(grid, z, q, dqdt, new_dt, cell, method, msg=gen_msg):
    dt_error(grid, z, q, dqdt, new_dt, cell, method, msg=gen_msg)
    
    
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

lf_constraints = \
"""
WARNING: The contents of `pop_constraints` will override the values of 
`pop_lf_Mstar`, `pop_lf_pstar`, and `pop_lf_alpha`. 
"""

def not_a_restart(prefix, has_burn):
    print("")
    print(line(separator))
    print(line("WARNING: This doesn't look like a restart:"))
    print(line("{!s}.chain.pkl is empty!".format(prefix)))
    
    if not has_burn:
        print(line("No burn-in data found. Continuing on as if from scratch."))
    else:
        print(line("Burn-in data found. Restarting from end of burn-in."))
        
    print(line(separator))    

def tau_tab_z_mismatch(igm, zmin_ok, zmax_ok, ztab):    
    print("")
    print(line(separator))
    print(line('WARNING: optical depth table shape mismatch (in redshift)'))
    print(line(separator))
    
    if type(igm.tabname) is dict:
        which = 'dict'
    else:
        which = 'tab'
        print(line("found       : {!s}".format(\
            igm.tabname[igm.tabname.rfind('/')+1:])))
    
    zmax_pop = min(igm.pf['pop_zform'], igm.pf['first_light_redshift'])
    
    print(line("zmin (pf)   : {0:g}".format(igm.pf['final_redshift'])))
    print(line("zmin ({0})  : {1:g}".format(which, ztab.min())))
    print(line("zmax (pf)   : {0:g}".format(zmax_pop)))
    print(line("zmax ({0})  : {1:g}".format(which, ztab.max())))

    if not zmin_ok:
        print(line(("this is OK  : we'll transition to an on-the-fly tau " +\
            "calculator at z={0:.2g}").format(ztab.min())))
        if (0 < igm.pf['EoR_xavg'] < 1):
            print(line(("            : or whenever x > {0:.1e}, whichever " +\
                "comes first").format(igm.pf['EoR_xavg'])))

    print(line(separator))
    print("")

def tau_tab_E_mismatch(pop, tabname, Emin_ok, Emax_ok, Etab):    
    print("")
    print(line(separator))
    print(line('WARNING: optical depth table shape mismatch (in photon ' +\
        'energy)'))    
    print(line(separator))        
    
    if type(tabname) is dict:
        which = 'dict'
    else:
        which = 'tab'
        print(line("found       : {!s}".format(\
            tabname[tabname.rfind('/')+1:])))
    
    print(line("Emin (pf)   : {0:g}".format(pop.pf['pop_Emin'])))
    print(line("Emin ({0})  : {1:g}".format(which, Etab.min())))
    print(line("Emax (pf)   : {0:g}".format(pop.pf['pop_Emax'])))
    print(line("Emax ({0})  : {1.g}".format(which, Etab.max())))

    if Etab.min() < pop.pf['pop_Emin']:
        print(line(("this is OK  : we'll discard E < {0:.2e} eV entries in " +\
            "table").format(pop.pf['pop_Emin'])))

    if Etab.max() > pop.pf['pop_Emax']:
        print(line(("this is OK  : we'll discard E > {0:.2e} eV entries in " +\
            "table").format(pop.pf['pop_Emax'])))

    print(line(separator))

def no_tau_table(urb):
    print("")
    print(line(separator))
    print(line('WARNING: no optical depth table found'))    
    print(line(separator))
    print(line("looking for : {!s}".format(urb.tabname)))
    if urb.pf['tau_prefix'] is not None:
        print(line("in          : {!s}".format(urb.pf['tau_prefix'])))
    elif ARES is not None:
        print(line("in          : {!s}/input/optical_depth".format(ARES)))
    else:
        print(line("in          : nowhere! set $ARES or tau_prefix"))

    print(line(separator))
    print(line("Generating a new table will take 5-10 minutes..."))

    print(line(separator))

def negative_SFRD(z, Tmin, fstar, dfcolldt, sfrd):
    print("")
    print(line(separator))
    print(line('ERROR (SFRD < 0)'))
    print(line(separator))
    print(line("z           : {0:.3g}".format(z)))
    print(line("Tmin        : {0:.3e}".format(Tmin)))
    print(line("fstar       : {0:.3e}".format(fstar)))
    print(line("dfcoll / dt : {0:.3e}".format(dfcolldt)))    
    print(line("SFRD        : {0:.3e}".format(sfrd)))
    print(line(separator))        

def tau_quad(igm):
    print("")
    print(line(separator))
    print(line('ERROR (SFRD < 0)'))
    print(line(separator))
    print(line("z           : {0:.3g}".format(z)))
    print(line("Tmin        : {0:.3e}".format(Tmin)))
    print(line("fstar       : {0:.3e}".format(fstar)))
    print(line("dfcoll / dt : {0:.3e}".format(dfcolldt)))    
    print(line("SFRD        : {0:.3e}".format(sfrd)))
    print(line(separator))
    
def missing_hmf_tab(hmf):
    print("")
    print(line(separator))
    print(line('WARNING: Could not find supplied hmf table.'))
    print(line(separator))    
    
    print(line('Was looking for:'))
    print(line(''))
    print(line('    {!s}'.format(hmf.pf['hmf_table'])))
    

    print(line(''))
    print(line('Will search for a suitable replacement in:'))
    print(line(''))
    print(line('    {!s}/input/hmf'.format(ARES)))
    print(line(''))
    print(line(separator))

def no_hmf(hmf):
    print("")
    print(line(separator))
    print(line('ERROR: Cannot generate halo mass function'))
    print(line(separator))

    if not have_ARES_env:
        s = \
        """
        It looks like you have not yet set the ARES environment variable, 
        which is needed to locate various input files. Make sure to source 
        your .bashrc or .cshrc (or equivalent) when finished! 
        """
    else:
        s = \
        """
        It looks like you have set the ARES environment variable. Is it 
        correct? Have you sourced your .bashrc or .cshrc (or equivalent) to 
        ensure that it is defined? 
        """

    if not (have_pycamb and have_hmf):
        s = \
        """
        If you've made no attempt to use non-default cosmological or HMF 
        parameters, it could just be that you forgot to run the remote.py script,
        which will download a default HMF lookup table.

        If you'd like to generate halo mass function lookup tables of your
        own, e.g., using fits other than the Sheth-Tormen form, or with 
        non-default cosmological parameters, you'll need to install hmf and
        pycamb.
        """
    
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')
    
    for l in snew_by_line:
        print(line(l))

        
    if not (have_pycamb and have_hmf):
        print(line(''))
        print(line('It looks like you\'re missing both hmf and pycamb.'))
    elif not have_pycamb:
        print(line(''))
        print(line('It looks like you\'re missing pycamb.'))
    elif not have_hmf:
        print(line(''))
        print(line('It looks like you\'re missing hmf.'))
    
    print(line(separator))


modelgrid_loadbalance = ""











    
        
