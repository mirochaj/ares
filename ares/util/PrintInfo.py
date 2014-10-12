"""

PrintInfo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 17 15:05:13 MDT 2014

Description: 

"""

import numpy as np
import types, os, textwrap
from .NormalizeSED import emission_bands
from ..physics.Constants import cm_per_kpc, m_H

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank; size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0; size = 1
 
# FORMATTING   
width = 84
pre = post = '#'*4    
twidth = width - len(pre) - len(post) - 2
#

ARES = os.environ.get('ARES')

e_methods = \
{
 0: 'all photo-electron energy -> heat',
 1: 'Shull & vanSteenberg (1985)',
 2: 'Ricotti, Gnedin, & Shull (2002)',
 3: 'Furlanetto & Stoever (2010)'
}
             
rate_srcs = \
{
 'fk94': 'Fukugita & Kawasaki (1994)',
 'chianti': 'Chianti'
}
             
S_methods = \
{
 1: 'Salpha = const. = 1',
 2: 'Chuzhoy, Alvarez, & Shapiro (2005)',
 3: 'Furlanetto & Pritchard (2006)'
}             

def line(s, just='l'):
    """ 
    Take a string, add a prefix and suffix (some number of # symbols).
    
    Optionally justify string, 'c' for 'center', 'l' for 'left', and 'r' for
    'right'. Defaults to left-justified.
    
    """
    if just == 'c':
        return "%s %s %s" % (pre, s.center(twidth), post)
    elif just == 'l':
        return "%s %s %s" % (pre, s.ljust(twidth), post)
    else:
        return "%s %s %s" % (pre, s.rjust(twidth), post)
        
def tabulate(data, rows, cols, cwidth=12, fmt='%.4e'):
    """
    Take table, row names, column names, and output nicely.
    """
    
    assert (cwidth % 2 == 0), \
        "Table elements must have an even number of characters."
        
    assert (len(pre) + len(post) + (1 + len(cols)) * cwidth) <= width, \
        "Table wider than maximum allowed width!"
    
    # Initialize empty list of correct length
    hdr = [' ' for i in range(width)]
    hdr[0:len(pre)] = list(pre)
    hdr[-len(post):] = list(post)
    
    hnames = []
    for i, col in enumerate(cols):
        tmp = col.center(cwidth)
        hnames.extend(list(tmp))
            
    start = len(pre) + cwidth + 3
    hdr[start:start + len(hnames)] = hnames
    
    # Convert from list to string        
    hdr_s = ''
    for element in hdr:
        hdr_s += element
        
    print hdr_s

    # Print out data
    for i in range(len(rows)):
    
        d = [' ' for j in range(width)]
        
        d[0:len(pre)] = list(pre)
        d[-len(post):] = list(post)
        
        d[len(pre)+1:len(pre)+1+len(rows[i])] = list(rows[i])
        d[len(pre)+1+cwidth] = ':'

        # Loop over columns
        numbers = ''
        for j in range(len(cols)):
            if type(data[i][j]) is str:
                numbers += data[i][j].center(cwidth)
                continue
            elif type(data[i][j]) is bool:
                numbers += str(int(data[i][j])).center(cwidth)
                continue 
            numbers += (fmt % data[i][j]).center(cwidth)
        numbers += ' '

        c = len(pre) + 1 + cwidth + 2
        d[c:c+len(numbers)] = list(numbers)
        
        d_s = ''
        for element in d:
            d_s += element
    
        print d_s
        
def print_warning(s, headerd='WARNING'):
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')
    
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    for l in snew_by_line:
        print line(l)
    
    print "#"*width        

def print_sim(sim):

    if rank > 0:
        return

    warnings = []

    header = 'Initializer: Radiative Transfer Simulation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    print line('-'*twidth)       
    print line('Book-Keeping')     
    print line('-'*twidth)
    
    if sim.pf['dtDataDump'] is not None:
        print line("dtDataDump  : every %i Myr" % sim.pf['dtDataDump'])
    else:
        print line("dtDataDump  : no regularly-spaced time dumps")
    
    if sim.pf['dzDataDump'] is not None:
        print line("dzDataDump  : every dz=%.2g" % sim.pf['dzDataDump'])
    else:
        print line("dzDataDump  : no regularly-spaced redshift dumps")    
       
    print line("initial dt  : %.2g Myr" % sim.pf['initial_timestep'])        
           
    rdt = ""
    for element in sim.pf['restricted_timestep']:
        rdt += '%s, ' % element
    rdt = rdt.strip().rstrip(',')       
    print line("restrict dt : %s" % rdt)
    print line("max change  : %.4g%% per time-step" % \
        (sim.pf['epsilon_dt'] * 100))

    print line('-'*twidth)       
    print line('Grid')     
    print line('-'*twidth)
    
    print line("cells       : %i" % sim.pf['grid_cells'], just='l')
    print line("logarithmic : %i" % sim.pf['logarithmic_grid'], just='l')
    print line("r0          : %.3g (code units)" % sim.pf['start_radius'], 
        just='l')
    print line("size        : %.3g (kpc)" \
        % (sim.pf['length_units'] / cm_per_kpc), just='l')
    print line("density     : %.2e (g cm**-3 / m_H)" % (sim.pf['density_units'] / m_H))
    
    print line('-'*twidth)       
    print line('Chemical Network')     
    print line('-'*twidth)
    
    Z = ''
    A = ''
    for i, element in enumerate(sim.pf['Z']):
        if element == 1:
            Z += 'H'
            A += '%.2g' % (sim.pf['abundances'][i])
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['abundances'][i])
            
    print line("elements    : %s" % Z, just='l')
    print line("abundances  : %s" % A, just='l')
    print line("rates       : %s" % rate_srcs[sim.pf['rate_source']], 
        just='l')
    
    print line('-'*twidth)       
    print line('Physics')     
    print line('-'*twidth)
    
    print line("radiation   : %i" % sim.pf['radiative_transfer'])
    print line("isothermal  : %i" % sim.pf['isothermal'], just='l')
    print line("expansion   : %i" % sim.pf['expansion'], just='l')
    if sim.pf['radiative_transfer']:
        print line("phot. cons. : %i" % sim.pf['photon_conserving'])
        print line("planar      : %s" % sim.pf['plane_parallel'], 
            just='l')        
    print line("electrons   : %s" % e_methods[sim.pf['secondary_ionization']], 
        just='l')
            
    # Should really loop over sources here        
    
    if sim.pf['radiative_transfer']:
    
        print line('-'*twidth)       
        print line('Source')     
        print line('-'*twidth)        
        
        print line("type        : %s" % sim.pf['source_type'])
        if sim.pf['source_type'] == 'star':
            print line("T_surf      : %.2e K" % sim.pf['source_temperature'])
            print line("Qdot        : %.2e photons / sec" % sim.pf['source_qdot'])
        
        print line('-'*twidth)       
        print line('Spectrum')     
        print line('-'*twidth)
        print line('not yet implemented')


        #if sim.pf['spectrum_E'] is not None:
        #    tabulate()
        

    print "#"*width
    print ""

def print_pop(pop):
    """
    Print information about a population to the screen.

    Parameters
    ----------
    pop : glorb.populations.*Population instance

    """

    if rank > 0 or not pop.pf['verbose']:
        return

    warnings = []

    alpha = pop.pf['spectrum_alpha']
    Emin = pop.pf['spectrum_Emin']
    Emax = pop.pf['spectrum_Emax']
    EminNorm = pop.pf['spectrum_EminNorm']
    EmaxNorm = pop.pf['spectrum_EmaxNorm']

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

    norm_by = pop.pf['norm_by']
    bands = emission_bands(Emin, Emax, pop.pf['xray_Emin']) 

    if len(bands) == 1:
        norm_by == bands[0]   

    if pop.pf['source_type'] == 'bh':
        header  = 'Initializer: BH Population'
    elif pop.pf['source_type'] == 'star':
        header  = 'Initializer: Stellar Population'

    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Redshift Evolution')
    print line('-'*twidth)

    # Redshift evolution stuff
    if pop.model <= 2:
        if pop.pf['sfrd'] is not None:
            print line("SF          : parameterized")
        else:
            if pop.pf['Mmin'] is None:
                print line("SF          : in halos w/ Tvir >= 10**%g K" \
                    % (round(np.log10(pop.pf['Tmin']), 2)))
            else:
                print line("SF          : in halos w/ M >= 10**%g Msun" \
                    % (round(np.log10(pop.pf['Mmin']), 2)))
            print line("HMF         : %s" % pop.pf['fitting_function'])
            print line("fstar       : %g" % pop.pf['fstar'])
        print line("model       : %i" % pop.model)

        if pop.model >= 0:
            print line("fbh         : 10**%g" % np.log10(pop.pf['fbh']))

    else:
        print "#### PopIII      : in halos w/ Tvir >= 10**%g K" \
            % (round(np.log10(pop.pf['Tmin'], 2)))
        print "#### HMF         : %s" % pop.pf['fitting_function']                    
        print "#### fstar       : %g" % pop.pf['fstar']
        print "####"            
        print "#### fbh         : %g" % pop.pf['fbh']
        print "#### fedd        : %g" % pop.pf['fedd']
        print "#### eta         : %g" % pop.pf['eta']

    ##
    # SPECTRUM STUFF
    ##
    print line('-'*twidth)
    print line('Spectrum')
    print line('-'*twidth)

    cols = ['lw', 'uv', 'xray']
    rows = ['is_src', 'approx RTE', 'Eavg / eV', 'erg / g', 'photons / b']
    data = [[bool(pop.pf['is_lya_src']), bool(pop.pf['is_ion_src_cgm']), 
        bool(pop.pf['is_xray_src'])]]
    data.append([bool(pop.pf['approx_lya']), 'n/a', bool(pop.pf['approx_xray'])])
    data.append((pop.Elw, pop.Eion, pop.Ex))
    data.append((pop.cLW, pop.cUV, pop.cX))
    data.append((pop.Nlw, pop.Nion, pop.Nx))

    tabulate(data, rows, cols)

    if not (pop.pf['approx_lya'] and pop.pf['approx_xray']):
        print line("sed         :  %s " % pop.pf['spectrum_type'])
        print line("sed         :  normalized by %s emission" % norm_by)

    if pop.pf['spectrum_type'] == 'bb':    
        print line("sed         :  T = 10**%.3g K" \
            % np.log10(pop.pf['source_temperature']))

    if pop.pf['spectrum_type'] in ['mcd', 'simpl']:
        print line("Mbh         :  %g " % pop.pf['source_mass'])

    if pop.pf['spectrum_type'] == 'simpl':
        print line("Gamma       :  %g " % pop.pf['spectrum_alpha'])
        print line("fsc         :  %g " % pop.pf['spectrum_fsc'])

    if np.isfinite(pop.pf['spectrum_logN']):
        print line("logN        :  %g " % pop.pf['spectrum_logN'])

    print "#"*width

    for warning in warnings:
        print_warning(warning)

def print_rb(rb):
    """
    Print information about radiation background calculation to screen.

    Parameters
    ----------
    igm : glorb.evolve.IGM instance
    zarr : np.ndarray
        Redshift points.
    xarr : np.ndarray
        Ionized fraction values at corresponding redshifts.

    """

    if rank > 0 or not rb.pf['verbose']:
        return

    if rb.pf['approx_xray']:
        return

    warnings = []        

    header = 'Initializer: Radiation Background'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Redshift & Energy Range')
    print line('-'*twidth)

    if rb.pf['redshift_bins'] is not None:
        print line("Emin (eV)         : %.1e" % rb.igm.E0)
        print line("Emax (eV)         : %.1e" % rb.igm.E1)

        if hasattr(rb.igm, 'z'):
            print line("zmin              : %.2g" % rb.igm.z.min())    
            print line("zmax              : %.2g" % rb.igm.z.max())    
            print line("redshift bins     : %i" % rb.igm.L)
            print line("frequency bins    : %i" % rb.igm.N)

        if hasattr(rb.igm, 'tabname'):

            if rb.igm.tabname is not None:
                print line('-'*twidth)
                print line('Tabulated IGM Optical Depth')
                print line('-'*twidth)

                fn = rb.igm.tabname[rb.igm.tabname.rfind('/')+1:]
                path = rb.igm.tabname[:rb.igm.tabname.rfind('/')+1]

                print line("file              : %s" % fn)

                if ARES in path:
                    path = path.replace(ARES, '')
                    print line("path              : $ARES%s" % path)
                else:
                    print line("path              : %s" % path)

    else:
        print line("Emin (eV)         : %.1e" % rb.pf['spectrum_Emin'])
        print line("Emax (eV)         : %.1e" % rb.pf['spectrum_Emax'])
        print line("NOTE              : this is a continuous radiation field!")

    print "#"*width

    for warning in warnings:
        print_warning(warning)

def print_sim(sim):
    """
    Print information about 21-cm simulation to screen.

    Parameters
    ----------
    sim : instance of Simulation class

    """

    if rank > 0 or not sim.pf['verbose']:
        return

    warnings = []

    header = 'Initializer: 21-cm Simulation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Book-Keeping')
    print line('-'*twidth)

    print line("z_initial   : %.1i" % sim.pf['initial_redshift'])
    if sim.pf['radiative_transfer']:
        print line("first-light : z=%.1i" % sim.pf['first_light_redshift'])
    if sim.pf['stop'] is not None:
        print line("z_final     : @ turning point %s " % sim.pf['stop'])
    else:
        if sim.pf['stop_xavg'] is not None:    
            print line("z_final     : when x_i > %.6g OR" % sim.pf['stop_xavg'])

        print line("z_final     : %.2g" % sim.pf['final_redshift'])

    if sim.pf['dtDataDump'] is not None:
        print line("dtDataDump  : every %i Myr" % sim.pf['dtDataDump'])
    else:
        print line("dtDataDump  : no regularly-spaced time dumps")

    if sim.pf['dzDataDump'] is not None:
        print line("dzDataDump  : every dz=%.2g" % sim.pf['dzDataDump'])
    else:
        print line("dzDataDump  : no regularly-spaced redshift dumps")    

    if sim.pf['max_dt'] is not None:  
        print line("max_dt      : %.2g Myr" % sim.pf['max_dt'])
    else:
        print line("max_dt      : no maximum time-step")

    if sim.pf['max_dz'] is not None:  
        print line("max_dz      : %.2g" % sim.pf['max_dz'])
    else:
        print line("max_dz      : no maximum redshift-step") 

    print line("initial dt  : %.2g Myr" % sim.pf['initial_timestep'])        

    rdt = ""
    for element in sim.pf['restricted_timestep']:
        rdt += '%s, ' % element
    rdt = rdt.strip().rstrip(',')       
    print line("restrict dt : %s" % rdt)
    print line("max change  : %.4g%% per time-step" % \
        (sim.pf['epsilon_dt'] * 100))

    ##
    # ICs
    ##
    if ARES and hasattr(sim, 'inits_path'):

        print line('-'*twidth)
        print line('Initial Conditions')
        print line('-'*twidth)

        fn = sim.inits_path[sim.inits_path.rfind('/')+1:]
        path = sim.inits_path[:sim.inits_path.rfind('/')+1]

        print line("file        : %s" % fn)

        if ARES in path:
            path = path.replace(ARES, '')
            print line("path        : $ARES%s" % path)
        else:
            print line("path        : %s" % path)

        if sim.pf['initial_redshift'] > sim.pf['first_light_redshift']:
            print line("FYI         : Can set initial_redshift=first_light_redshift for speed-up.", 
                just='l')

    ##
    # PHYSICS
    ##        

    print line('-'*twidth)
    print line('Physics')
    print line('-'*twidth)

    print line("radiation   : %i" % sim.pf['radiative_transfer'])
    print line("electrons   : %s" % e_methods[sim.pf['secondary_ionization']])
    if type(sim.pf['clumping_factor']) is types.FunctionType:
        print line("clumping    : parameterized")
    else:  
        print line("clumping    : C = const. = %i" % sim.pf['clumping_factor'])

    if type(sim.pf['feedback']) in [int, bool]:
        print line("feedback    : %i" % sim.pf['feedback'])
    else:
        print line("feedback    : %i" % sum(sim.pf['feedback']))

    Z = ''
    A = ''
    for i, element in enumerate(sim.pf['Z']):
        if element == 1:
            Z += 'H'
            A += '%.2g' % (sim.pf['abundances'][i])
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['abundances'][i])

    print line("elements    : %s" % Z, just='l')
    print line("abundance   : %s" % A, just='l')
    print line("approx He   : %i" % sim.pf['approx_helium'])
    print line("rates       : %s" % rate_srcs[sim.pf['rate_source']], 
        just='l')

    print line("approx Sa   : %s" % S_methods[sim.pf['approx_Salpha']], 
        just='l')

    print "#"*width

    #if not ARES:
    #    warnings.append(hmf_no_tab)
    #elif not os.path.exists('%s/input/hmf' % ARES):
    #    warnings.append(hmf_no_tab)

    for warning in warnings:
        print_warning(warning)       

def print_fit(fit, steps, burn=0):         

    if rank > 0:
        return

    warnings = []

    header = 'Initializer: Parameter Estimation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)       
    print line('Measurement to be Fit')     
    print line('-'*twidth)

    if not hasattr(fit, "chain"):
        cols = ['position', 'error']
    else:
        cols = ['position', 'std-dev']   
        print line('Using supplied MCMC chain rather than input model.')
        print line('-'*twidth)

    i = 0
    rows = []
    data = []
    for i, element in enumerate(fit.measurement_map):

        tp, val = element

        if tp == 'trans':
            continue

        if val == 0:
            rows.append('z_%s' % tp)
        else:
            rows.append('T_%s' % tp)

        if not hasattr(fit, "chain"):
            data.append([fit.mu[i], fit.error[i]])
        else:
            iML = np.argmax(fit.logL)
            data.append([fit.chain[iML,i], np.std(fit.chain[:,i])])

    tabulate(data, rows, cols)    

    print line('-'*twidth)       
    print line('Parameter Space')     
    print line('-'*twidth)

    data = []    
    cols = ['prior_dist', 'prior_p1', 'prior_p2']
    rows = fit.parameters    
    for i, row in enumerate(rows):

        if row in fit.priors:
            tmp = [fit.priors[row][0]]
            tmp.extend(fit.priors[row][1:])
        else:
            tmp = ['n/a'] * 3

        data.append(tmp)

    tabulate(data, rows, cols, fmt='%.2g')

    print line('-'*twidth)       
    print line('Exploration')     
    print line('-'*twidth)


    print line("nprocs      : %i" % size)
    print line("nwalkers    : %i" % fit.nwalkers)
    print line("burn-in     : %i" % burn)
    print line("steps       : %i" % steps)
    print line("outputs     : %s*.pkl" % fit.prefix)

    print "#"*width
    print ""

def print_model_grid():

    header = 'Model Grid'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Input Model')     
    print line('-'*twidth)

def print_constraint():
    pass










