"""

PrintInfo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 17 15:05:13 MDT 2014

Description: 

"""

import numpy as np
from types import FunctionType
import types, os, textwrap, glob, re
from .NormalizeSED import emission_bands
from ..physics.Constants import cm_per_kpc, m_H, nu_0_mhz, g_per_msun, s_per_yr

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
 
class ErrorIgnore(object):
   def __init__(self, errors, errorreturn = None, errorcall = None):
      self.errors = errors
      self.errorreturn = errorreturn
      self.errorcall = errorcall

   def __call__(self, function):
      def returnfunction(*args, **kwargs):
         try:
            return function(*args, **kwargs)
         except Exception as E:
            if type(E) not in self.errors:
               raise E
            if self.errorcall is not None:
               self.errorcall(E, *args, **kwargs)
            return self.errorreturn
      return returnfunction 

# FORMATTING   
width = 74
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
        
def print_warning(s, header='WARNING'):
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')
    
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width
    
    for l in snew_by_line:
        print line(l)
    
    print "#"*width        

def print_1d_sim(sim):

    if rank > 0:
        return

    warnings = []

    header = 'Radiative Transfer Simulation'
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
    print line("density     : %.2e (H atoms cm**-3)" % (sim.pf['density_units']))
    
    print line('-'*twidth)       
    print line('Chemical Network')     
    print line('-'*twidth)
    
    Z = ''
    A = ''
    for i, element in enumerate(sim.grid.Z):
        if element == 1:
            Z += 'H'
            A += '%.2g' % (1)
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['helium_by_number'])
            
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

def print_rate_int(tab):
    """
    Print information about a population to the screen.

    Parameters
    ----------
    pop : ares.populations.*Population instance

    """
    
    if rank > 0 or not pop.pf['verbose']:
        return

    warnings = []

    header  = 'Tabulated Rate Coefficient Integrals'
    
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    #print line('-'*twidth)
    #print line('Redshift Evolution')
    #print line('-'*twidth)    
    #
    print "#"*width

    for warning in warnings:
        print_warning(warning)

#@ErrorIgnore(errors=[KeyError])
def print_pop(pop):
    """
    Print information about a population to the screen.

    Parameters
    ----------
    pop : ares.populations.*Population instance

    """

    if rank > 0 or not pop.pf['verbose']:
        return

    warnings = []

    alpha = pop.pf['pop_alpha']
    Emin = pop.pf['pop_Emin']
    Emax = pop.pf['pop_Emax']
    EminNorm = pop.pf['pop_EminNorm']
    EmaxNorm = pop.pf['pop_EmaxNorm']

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

    #norm_by = pop.pf['norm_by']
    #bands = emission_bands(Emin, Emax, pop.pf['xray_Emin']) 
    #
    #if len(bands) == 1:
    #    norm_by == bands[0]   

    if pop.pf['pop_type'] == 'bh':
        header  = 'BH Population'
    elif pop.pf['pop_type'] == 'star':
        header  = 'Stellar Population'
    elif pop.pf['pop_type'] == 'galaxy':
        header  = 'Galaxy Population'        

    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Redshift Evolution')
    print line('-'*twidth)

    # Redshift evolution stuff
    if pop.pf['pop_sfrd'] is not None:
        if type(pop.pf['pop_sfrd']) is str:
            print line("SFRD        : %s" % pop.pf['pop_sfrd'])
        else:
            print line("SFRD        : parameterized")
    elif pop.pf['pop_rhoL'] is not None:
        if type(pop.pf['pop_rhoL']) is FunctionType:
            print line("rho_L(z)    : parameterized")
        else:
            print line("rho_L(z)    : %s" % pop.pf['pop_rhoL'])
    else:
        if pop.pf['pop_Mmin'] is None:
            print line("SF          : in halos w/ Tvir >= 10**%g K" \
                % (round(np.log10(pop.pf['pop_Tmin']), 2)))
        else:
            print line("SF          : in halos w/ M >= 10**%g Msun" \
                % (round(np.log10(pop.pf['pop_Mmin']), 2)))
        print line("HMF         : %s" % pop.pf['hmf_func'])
        print line("fstar       : %g" % pop.pf['pop_fstar'])
    
    print line('-'*twidth)
    print line('Radiative Output')
    print line('-'*twidth)
    
    print line("yield (erg / s / SFR) : %g" \
        % (pop.yield_per_sfr * g_per_msun / s_per_yr))
    print line("EminNorm (eV)         : %g" % (pop.pf['pop_EminNorm']))
    print line("EmaxNorm (eV)         : %g" % (pop.pf['pop_EmaxNorm']))    

    ##
    # SPECTRUM STUFF
    ##
    print line('-'*twidth)
    print line('Spectrum')
    print line('-'*twidth)
    
    print line("SED               : %s" % (pop.pf['pop_sed']))
    print line("Emin (eV)         : %g" % (pop.pf['pop_Emin']))
    print line("Emax (eV)         : %g" % (pop.pf['pop_Emax']))

    if pop.pf['pop_sed'] == 'pl':
        print line("alpha             : %g" % pop.pf['pop_alpha'])
        print line("logN              : %g" % pop.pf['pop_logN'])
    elif pop.pf['pop_sed'] == 'mcd':
        print line("mass (Msun)       : %g" % pop.pf['pop_mass'])
        print line("rmax (Rg)         : %g" % pop.pf['pop_rmax'])
    else:
        print line("from source       : %s" % pop.pf['pop_sed'])
        
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

    if rb.pf['approx_xrb']:
        return

    warnings = []        

    header = 'Radiation Background'
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

                if type(rb.igm.tabname) is dict:
                    print line("file              : actually, a dictionary via tau_table")
                else:
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
        
        if rb.pf['spectrum_Emin'] < 13.6:
            if not rb.pf['discrete_lwb']:
                print line("NOTE              : this is a continuous radiation field!")
            else:
                print line("NOTE              : discretized over first %i Ly-n bands" % rb.pf['lya_nmax'])
        else:
            print line("NOTE              : this is a continuous radiation field!")

    print "#"*width

    for warning in warnings:
        print_warning(warning)

def print_21cm_sim(sim):
    """
    Print information about 21-cm simulation to screen.

    Parameters
    ----------
    sim : instance of Simulation class

    """

    if rank > 0 or not sim.pf['verbose']:
        return

    warnings = []

    header = '21-cm Simulation'
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
    for i, element in enumerate(sim.grid.Z):
        if element == 1:
            Z += 'H'
            A += '%.2g' % 1.
        elif element == 2:
            Z += ', He'
            A += ', %.2g' % (sim.pf['helium_by_number'])

    print line("elements    : %s" % Z, just='l')
    print line("abundance   : %s" % A, just='l')
    print line("approx He   : %i" % sim.pf['approx_He'])
    print line("rates       : %s" % rate_srcs[sim.pf['rate_source']], 
        just='l')

    print line("approx Sa   : %s" % S_methods[sim.pf['approx_Salpha']], 
        just='l')

    print "#"*width

    for warning in warnings:
        print_warning(warning)       

def print_fit(fit, steps, burn=0, fit_TP=True):         

    if rank > 0:
        return

    warnings = []
    
    is_cov = True
    if len(fit.error.shape) == 1:
        is_cov = False

    header = 'Parameter Estimation'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    if is_cov:
        cols = ['position', 'error (diagonal of cov)']
    else:
        cols = ['position', 'error']   
        
    if fit_TP:

        print line('-'*twidth)       
        print line('Measurement to be Fit')     
        print line('-'*twidth)

        i = 0
        rows = []
        data = []
        for i, element in enumerate(fit.measurement_map):
        
            tp, val = element
        
            if tp == 'trans':
                continue

            if val == 0:
                if fit.measurement_units[0] == 'MHz':
                    rows.append('nu_%s (MHz)' % tp)
                else:
                    rows.append('z_%s' % tp)
            else:
                rows.append('T_%s (mK)' % tp)

            unit = fit.measurement_units[val]
        
            if is_cov:
                col1, col2 = fit.mu[i], np.sqrt(np.diag(fit.error)[i])
            else:
                col1, col2 = fit.mu[i], fit.error[i]
                
            data.append([col1, col2])

        tabulate(data, rows, cols, cwidth=18)    

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

    tabulate(data, rows, cols, fmt='%.2g', cwidth=18)

    print line('-'*twidth)       
    print line('Exploration')     
    print line('-'*twidth)

    print line("nprocs      : %i" % size)
    print line("nwalkers    : %i" % fit.nwalkers)
    print line("burn-in     : %i" % burn)
    print line("steps       : %i" % steps)
    print line("outputs     : %s.*.pkl" % fit.prefix)
    
    print line('-'*twidth)       
    print line('Inline Analysis')     
    print line('-'*twidth)
    
    Nb = len(fit.blob_names)
    Nz = len(fit.blob_redshifts)
    perwalkerperstep = Nb * Nz * 8 
    MB = perwalkerperstep * fit.nwalkers * steps / 1e6

    print line("N blobs     : %i" % Nb)
    print line("N redshifts : %i" % Nz)
    print line("blob rate   : %i bytes / walker / step" % perwalkerperstep)
    print line("blob size   : %.2g MB (total)" % MB)

    print "#"*width
    print ""

def print_model_grid():
    if rank > 0:
        return
        
    header = 'Model Grid'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Input Model')     
    print line('-'*twidth)

def print_model_set(mset):
    if rank > 0:
        return
        
    header = 'Analysis: Model Set'
    print "\n" + "#"*width
    print "%s %s %s" % (pre, header.center(twidth), post)
    print "#"*width

    print line('-'*twidth)
    print line('Basic Information')     
    print line('-'*twidth)

    i = mset.prefix.rfind('/') # forward slash index
    
    # This means we're sitting in the right directory already
    if i == - 1:
        path = './'
        prefix = mset.prefix
    else:
        path = mset.prefix[0:i+1]
        prefix = mset.prefix[i+1:]

    print line("path        : %s" % path)    
    print line("prefix      : %s" % prefix)
    print line("N-d         : %i" % len(mset.parameters))

    print line('-'*twidth)
    for i, par in enumerate(mset.parameters):
        print line("param    #%s: %s" % (str(i).zfill(2), par))

    print line('-'*twidth)
    print line('Data Available (filename = prefix + .suffix*.pkl)')
    print line('-'*twidth)

    suffixes = []
    for fn in glob.glob('%s%s*' % (path, prefix)):

        if not re.search('.pkl', fn):
            continue

        suffix = fn.split('.')[1]

        if suffix not in suffixes:
            suffixes.append(suffix)

    for i, suffix in enumerate(suffixes):
        print line("suffix   #%s: %s" % (str(i).zfill(2), suffix))
        
    print line('-'*twidth)
    print line('Data Products (filename = prefix + .suffix*.hdf5)')
    print line('-'*twidth)    
    
    suffixes = []
    for fn in glob.glob('%s%s*' % (path, prefix)):

        if not re.search('.hdf5', fn):
            continue

        suffix = fn.split('.')[1]

        if suffix not in suffixes:
            suffixes.append(suffix)

    for i, suffix in enumerate(suffixes):
        print line("suffix   #%s: %s" % (str(i).zfill(2), suffix))
        
    print "#"*width
    print ""
    








