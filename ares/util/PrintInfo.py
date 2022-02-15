"""

PrintInfo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jul 17 15:05:13 MDT 2014

Description:

"""

import os
import numpy as np
from ..data import ARES
from types import FunctionType
import types, os, textwrap, glob, re
from ..physics.Constants import cm_per_kpc, m_H, nu_0_mhz, g_per_msun, s_per_yr
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1


settings = {'width': 76, 'border': 2, 'pad': 1, 'col': 6}

HOME = os.environ.get('HOME')
if os.path.exists('{!s}/.ares/printout'.format(HOME)):
    col1, col2 = np.loadtxt('{!s}/.ares/printout'.format(HOME), unpack=True,
        dtype=str)

    for i, row in enumerate(col1):
        settings[col1[i].strip()] = int(col2[i].strip())

# FORMATTING
width = settings['width']
pre = post = '#' * settings['border']
twidth = width - len(pre) - len(post) - 2
pad = settings['pad']
#

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

def footer():
    print("#" * width)
    print("")

def header(s):
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, s.center(twidth), post))
    print("#" * width)

def separator():
    print(line('-' * twidth))

def line(s, just='l'):
    """
    Take a string, add a prefix and suffix (some number of # symbols).

    Optionally justify string, 'c' for 'center', 'l' for 'left', and 'r' for
    'right'. Defaults to left-justified.

    """
    if just == 'c':
        return "{0!s} {1!s} {2!s}".format(pre, s.center(twidth), post)
    elif just == 'l':
        return "{0!s} {1!s} {2!s}".format(pre, s.ljust(twidth), post)
    else:
        return "{0!s} {1!s} {2!s}".format(pre, s.rjust(twidth), post)

def tabulate(data, rows, cols, cwidth=12, fmt='{:.4e}'):
    """
    Take table, row names, column names, and output nicely.
    """

    if type(cwidth) == int:
        assert (cwidth % 2 == 0), \
            "Table elements must have an even number of characters."

        cwidth = [cwidth] * (len(cols) + 1)

    else:
        assert len(cwidth) == len(cols) + 1

    #assert (len(pre) + len(post) + (1 + len(cols)) * cwidth) <= width, \
    #    "Table wider than maximum allowed width!"

    # Initialize empty list of correct length
    hdr = [' ' for i in range(width)]
    hdr[0:len(pre)] = list(pre)
    hdr[-len(post):] = list(post)

    hnames = []
    for i, col in enumerate(cols):
        tmp = col.center(cwidth[i+1])
        hnames.extend(list(tmp))

    start = len(pre) + cwidth[0] + settings['pad']

    hdr[start:start + len(hnames)] = hnames

    # Convert from list to string
    hdr_s = ''
    for element in hdr:
        hdr_s += element

    print(hdr_s)

    # Print out data
    for i in range(len(rows)):

        d = [' ' for j in range(width)]

        d[0:len(pre)] = list(pre)
        d[-len(post):] = list(post)

        d[len(pre)+1:len(pre)+1+len(rows[i])] = list(rows[i])
        d[len(pre)+1+cwidth[0]] = ':'

        # Loop over columns
        numbers = ''
        for j in range(len(cols)):
            if isinstance(data[i][j], basestring):
                numbers += data[i][j].center(cwidth[j+1])
                continue
            elif type(data[i][j]) is bool:
                numbers += str(int(data[i][j])).center(cwidth[j+1])
                continue
            numbers += (fmt.format(data[i][j])).center(cwidth[j+1])
        numbers += ' '

        c = len(pre) + 1 + cwidth[0] + settings['pad']
        d[c:c+len(numbers)] = list(numbers)

        d_s = ''
        for element in d:
            d_s += element

        print(d_s)

def print_warning(s, header='WARNING'):
    dedented_s = textwrap.dedent(s).strip()
    snew = textwrap.fill(dedented_s, width=twidth)
    snew_by_line = snew.split('\n')

    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    for l in snew_by_line:
        print(line(l))

    print("#" * width)

def print_1d_sim(sim):

    if rank > 0:
        return

    warnings = []

    header = 'Radiative Transfer Simulation'
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    print(line('-' * twidth))
    print(line('Book-Keeping'))
    print(line('-' * twidth))

    if sim.pf['dtDataDump'] is not None:
        print(line("dtDataDump  : every {} Myr".format(sim.pf['dtDataDump'])))
    else:
        print(line("dtDataDump  : no regularly-spaced time dumps"))

    if sim.pf['dzDataDump'] is not None:
        print(line("dzDataDump  : every dz={0:.2g}".format(\
            sim.pf['dzDataDump'])))
    else:
        print(line("dzDataDump  : no regularly-spaced redshift dumps"))

    print(line("initial dt  : {0:.2g} Myr".format(sim.pf['initial_timestep'])))

    rdt = ""
    for element in sim.pf['restricted_timestep']:
        rdt += '{!s}, '.format(element)
    rdt = rdt.strip().rstrip(',')
    print(line("restrict dt : {!s}".format(rdt)))
    print(line("max change  : {0:.4g}% per time-step".format(\
        sim.pf['epsilon_dt'] * 100)))

    print(line('-' * twidth))
    print(line('Grid'))
    print(line('-' * twidth))

    print(line("cells       : {}".format(sim.pf['grid_cells']), just='l'))
    print(line("logarithmic : {}".format(sim.pf['logarithmic_grid']), just='l'))
    print(line("r0          : {0:.3g} (code units)".format(\
        sim.pf['start_radius']), just='l'))
    print(line("size        : {0:.3g} (kpc)".format(\
        sim.pf['length_units'] / cm_per_kpc), just='l'))
    print(line("density     : {0:.2e} (H atoms cm**-3)".format(\
        sim.pf['density_units'])))

    print(line('-' * twidth))
    print(line('Chemical Network'))
    print(line('-' * twidth))

    Z = ''
    A = ''
    for i, element in enumerate(sim.grid.Z):
        if element == 1:
            Z += 'H'
            A += '{0:.2g}'.format(1)
        elif element == 2:
            Z += ', He'
            A += ', {0:.2g}'.format(sim.pf['helium_by_number'])

    print(line("elements    : {!s}".format(Z), just='l'))
    print(line("abundances  : {!s}".format(A), just='l'))
    print(line("rates       : {!s}".format(rate_srcs[sim.pf['rate_source']]),
        just='l'))

    print(line('-'*twidth))
    print(line('Physics'))
    print(line('-'*twidth))

    print(line("radiation   : {}".format(sim.pf['radiative_transfer'])))
    print(line("isothermal  : {}".format(sim.pf['isothermal']), just='l'))
    print(line("expansion   : {}".format(sim.pf['expansion']), just='l'))
    if sim.pf['radiative_transfer']:
        print(line("phot. cons. : {}".format(sim.pf['photon_conserving'])))
        print(line("planar      : {!s}".format(sim.pf['plane_parallel']),
            just='l'))
    print(line("electrons   : {!s}".format(\
        e_methods[sim.pf['secondary_ionization']]), just='l'))

    # Should really loop over sources here

    if sim.pf['radiative_transfer']:

        print(line('-' * twidth))
        print(line('Source'))
        print(line('-' * twidth))

        print(line("type        : {!s}".format(sim.pf['source_type'])))
        if sim.pf['source_type'] == 'star':
            print(line("T_surf      : {0:.2e} K".format(\
                sim.pf['source_temperature'])))
            print(line("Qdot        : {0:.2e} photons / sec".format(\
                sim.pf['source_qdot'])))

        print(line('-' * twidth))
        print(line('Spectrum'))
        print(line('-' * twidth))
        print(line('not yet implemented'))

        #if sim.pf['spectrum_E'] is not None:
        #    tabulate()

    print("#" * width)
    print("")

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

    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    #print(line('-' * twidth))
    #print(line('Redshift Evolution'))
    #print(line('-' * twidth))
    #
    print("#" * width)

    for warning in warnings:
        print_warning(warning)

def print_hmf(hmf):
    header = 'Halo Mass function'
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    print(line('-' * twidth))
    print(line('Underlying Model'))
    print(line('-' * twidth))
    print(line("fittin function       : {0!s}".format(hmf.pf['hmf_model'])))
    if hmf.pf['hmf_wdm_mass'] is not None:
        print(line("wdm_mass              : {0:g}".format(hmf.pf['hmf_wdm_mass'])))

    print(line('-' * twidth))
    print(line('Table Limits & Resolution'))
    print(line('-' * twidth))

    if hmf.pf['hmf_dt'] is None:
        print(line("zmin                  : {0:g}".format(hmf.pf['hmf_zmin'])))
        print(line("zmax                  : {0:g}".format(hmf.pf['hmf_zmax'])))
        print(line("dz                    : {0:g}".format(hmf.pf['hmf_dz'])))
    else:
        print(line("tmin (Myr)            : {0:g}".format(hmf.pf['hmf_tmin'])))
        print(line("tmax (Myr)            : {0:g}".format(hmf.pf['hmf_tmax'])))
        print(line("dt   (Myr)            : {0:g}".format(hmf.pf['hmf_dt'])))

    print(line("Mmin (Msun)           : {0:e}".format(\
        10 ** hmf.pf['hmf_logMmin'])))
    print(line("Mmax (Msun)           : {0:e}".format(\
        10 ** hmf.pf['hmf_logMmax'])))
    print(line("dlogM                 : {0:g}".format(hmf.pf['hmf_dlogM'])))

    print("#" * width)

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

    header = 'ARES Population: Summary'
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    print(line('-' * twidth))
    print(line('Star Formation'))
    print(line('-' * twidth))

    # Redshift evolution stuff
    if pop.pf['pop_sfrd'] is not None:
        if isinstance(pop.pf['pop_sfrd'], basestring):
            print(line("SFRD        : {!s}".format(pop.pf['pop_sfrd'])))
        else:
            print(line("SFRD        : parameterized"))
    else:
        if pop.pf['pop_Mmin'] is None:
            print(line("SF          : in halos w/ Tvir >= 10**{0:g} K".format(\
                round(np.log10(pop.pf['pop_Tmin']), 2))))
        else:
            print(line(("SF          : in halos w/ M >= 10**{0:g} " +\
                "Msun").format(round(np.log10(pop.pf['pop_Mmin']), 2))))
        print(line("HMF         : {!s}".format(pop.pf['hmf_model'])))
        print(line("MAR scatter : {!s} dex".format(pop.pf['pop_scatter_mar'])))

    # Parameterized halo properties
    if pop.pf.Npqs > 0:
        if pop.pf.Npqs > 1:
            sf = lambda x: '[{}]'.format(x)
        else:
            sf = lambda x: ''

        for i, par in enumerate(pop.pf.pqs):

            pname = par.replace('pop_', '').ljust(9)

            s = pop.pf['pq_func{!s}'.format(sf(i))]

            if 'pq_faux{!s}'.format(sf(i)) not in pop.pf:
                print(line("{0!s}   : {1!s}".format(pname, s)))
                continue

            if pop.pf['pq_faux{!s}'.format(sf(i))] is not None:
                if pop.pf['pq_faux_meth{!s}'.format(sf(i))] == 'add':
                    s += ' + {!s}'.format(pop.pf['pq_faux{!s}'.format(sf(i))])
                else:
                    s += ' * {!s}'.format(pop.pf['pq_faux{!s}'.format(sf(i))])

            print(line("{0!s}: {1!s}".format(pname, s)))

    ##
    # SPECTRUM STUFF
    ##
    print(line('-' * twidth))
    print(line('Spectrum'))
    print(line('-' * twidth))

    print(line("SED         : {!s}".format(pop.pf['pop_sed'])))

    if pop.pf['pop_sed'] == 'pl':
        print(line("alpha       : {0:g}".format(\
            pop.pf['pop_alpha'])))
        print(line("logN        : {0:g}".format(pop.pf['pop_logN'])))
    elif pop.pf['pop_sed'] == 'mcd':
        print(line("mass (Msun) : {0:g}".format(pop.pf['pop_mass'])))
        print(line("rmax (Rg)   : {0:g}".format(pop.pf['pop_rmax'])))
    elif pop.pf['pop_sed'] in ['eldridge2009', 'leitherer1999']:
        print(line("Z           : {0:g}".format(pop.pf['pop_Z'])))
        print(line("IMF         : {0:g}".format(pop.pf['pop_imf'])))
        print(line("binaries?   : {0:g}".format(pop.pf['pop_binaries'])))
        print(line("burst       : {0:g}".format(pop.pf['pop_ssp'])))
        print(line("aging       : {0:g}".format(pop.pf['pop_aging'])))

    ##
    # Dust stuff

    # If SED not tabulated, print out radiative output.
    if pop.pf['pop_sed'] not in ['eldridge2009', 'leitherer1999']:
        print(line('-' * twidth))
        print(line('Radiative Output'))
        print(line('-' * twidth))
        if hasattr(pop, 'yield_per_sfr'):
            print(line("yield (erg / s / SFR) : {0:g}".format(\
                pop.yield_per_sfr * g_per_msun / s_per_yr)))

        print(line("Emin (eV)             : {0:g}".format(pop.pf['pop_Emin'])))
        print(line("Emax (eV)             : {0:g}".format(pop.pf['pop_Emax'])))
        print(line("EminNorm (eV)         : {0:g}".format(pop.pf['pop_EminNorm'])))
        print(line("EmaxNorm (eV)         : {0:g}".format(pop.pf['pop_EmaxNorm'])))

    ##
    # NOTES!
    print(line('-' * twidth))
    print(line('Special Notes'))
    print(line('-' * twidth))
    if pop.pf['pop_calib_lum'] is not None:
        s1 = "+ pop_calib_lum != None, which means".format(i)
        s1 += ' changes to pop_Z will *not* affect UVLF.'
        s2 = '  Set pop_calib_lum=None to restore "normal" behavior'
        s2 += ' (see S3.4 in Mirocha et al. 2017).'
        print(line(s1))
        print(line(s2))
    if pop.pf['pop_sfr_model'] == 'ensemble':
        s1 = "+ pop_sfr_model == 'ensemble', which means".format(i)
        s1 += ' luminosity at all wavelengths is determined via spectral'
        s2 = '  synthesis, which can be slow. Set pop_sed_degrade > 0'
        s2 += ' [in Angstroms] for speed-up.'
        print(line(s1))
        print(line(s2))

    # Other noteworthy things?

    if warnings != []:
        print(line('-' * twidth))
        print(line('Warnings'))
        print(line('-' * twidth))
        for warning in warnings:
            print_warning(warning)

    print("#" * width)

def _rad_type(sim, fluctuations=False):
    rows = []
    cols = ['sfrd', 'sed', 'radio', 'O/IR', 'Lya', 'LW', 'LyC', 'Xray', 'RTE']
    data = []
    for i, pop in enumerate(sim.pops):
        rows.append('pop #%i' % i)
        if re.search('link', pop.pf['pop_sfr_model']):
            junk, quantity, num = pop.pf['pop_sfr_model'].split(':')
            mod = '%s->%i' % (quantity, int(num))
        else:
            mod = pop.pf['pop_sfr_model']

        tmp = [mod, 'yes' if pop.pf['pop_sed_model'] else 'no']

        suffix = ['', '']
        for j, fl in enumerate([True, False]):
            if fl != fluctuations:
                continue

            for band in ['radio', 'oir', 'lya', 'lw', 'ion', 'heat']:
                is_src = pop.__getattribute__('is_src_%s%s' % (band, suffix[j]))

                if is_src:
                    tmp.append('x')
                else:
                    tmp.append('-')

            # No analog for RTE solution for fluctuations (yet)
            if fl:
                tmp.append(' ')

            if pop.pf['pop_solve_rte']:
                tmp.append('x')
            else:
                tmp.append('-')

        data.append(tmp)

    return data, rows, cols

def print_sim(sim, mgb=False):
    """
    Print information about simulation to screen.

    Parameters
    ----------
    sim : ares.simulations.Global21cm or PowerSpectrum21cm instance.

    """

    if rank > 0 or not sim.pf['verbose']:
        return

    # Poke sim.pops just to get loads in before print-out
    _pops = sim.pops

    header = 'ARES Simulation: Overview'
    print("\n" + "#"*width)
    print("%s %s %s" % (pre, header.center(twidth), post))
    print("#"*width)

    # Check for phenomenological models
    if sim.is_phenom:
        print("Phenomenological model! Not much to report...")
        print("#"*width)
        return

    cw =print(line('-'*twidth))
    print(line('Source Populations'))
    print(line('-'*twidth))

    data, rows, cols = _rad_type(sim)

    cw = settings['col']
    cwidth = [cw+1, cw+4] + [cw] * 8
    tabulate(data, rows, cols, cwidth=cwidth, fmt='{!s}')

    #print line('-'*twidth)
    #print line('Fluctuating Backgrounds')
    #print line('-'*twidth)
    #cw =
    #data, rows, cols = _rad_type(sim, fluctuations=True)
    #tabulate(data, rows, cols, cwidth=[8,12,8,8,8,8,8,8,8,8], fmt='{!s}')

    if mgb:
        print("#" * width)
        return

    ct = 0
    for i, pop in enumerate(sim.pops):
        if pop.pf['pop_calib_lum'] is not None:
            if ct == 0:
                print(line('-' * twidth))
                print(line('Notes'))
                print(line('-' * twidth))

            s1a = "+ pop_calib_lum != None, which means changes to pop_Z".format(i)
            s1b = '  will *not* affect UVLF. Set pop_calib_lum=None to restore'
            s1c = '  "normal" behavior (see S3.4 in Mirocha et al. 2017).'
            print(line(s1a))
            print(line(s1b))
            print(line(s1c))
            ct += 1

        # Other noteworthy things?


    print(line('-' * twidth))
    print(line('Physics'))
    print(line('-' * twidth))

    phys_pars = ['cgm_initial_temperature', 'clumping_factor',
        'secondary_ionization', 'approx_Salpha', 'include_He',
        'feedback_LW', 'feedback_LW_Mmin', 'feedback_LW_fsh']

    cosm_pars = ["omega_m_0", "omega_b_0", "omega_l_0", "hubble_0",
        "helium_by_number", "sigma_8"]

    for par in phys_pars:
        val = sim.pf[par]


        if ('feedback_LW' in par) and (par != 'feedback_LW'):
            if not sim.pf['feedback_LW']:
                continue

        if val is None:
            print(line('{!s} : None'.format(par.ljust(30))))
        elif type(val) in [list, tuple]:
            print(line('{0!s} : {1!s}'.format(par.ljust(30), val)))
        elif type(val) in [int, float]:
            print(line('{0!s} : {1:g}'.format(par.ljust(30), val)))
        else:
            print(line('{0!s} : {1!s}'.format(par.ljust(30), val)))

    print("#" * width)


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
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    print(line('-' * twidth))
    print(line('Redshift & Energy Range'))
    print(line('-' * twidth))

    if rb.pf['redshift_bins'] is not None:
        print(line("Emin (eV)         : {0:.1e}".format(rb.igm.E0)))
        print(line("Emax (eV)         : {0:.1e}".format(rb.igm.E1)))

        if hasattr(rb.igm, 'z'):
            print(line("zmin              : {0:.2g}".format(rb.igm.z.min())))
            print(line("zmax              : {0:.2g}".format(rb.igm.z.max())))
            print(line("redshift bins     : {}".format(rb.igm.L)))
            print(line("frequency bins    : {}".format(rb.igm.N)))

        if hasattr(rb.igm, 'tabname'):

            if rb.igm.tabname is not None:
                print(line('-' * twidth))
                print(line('Tabulated IGM Optical Depth'))
                print(line('-' * twidth))

                if type(rb.igm.tabname) is dict:
                    print(line("file              : actually, a dictionary " +\
                        "via tau_table"))
                else:
                    fn = rb.igm.tabname[rb.igm.tabname.rfind('/')+1:]
                    path = rb.igm.tabname[:rb.igm.tabname.rfind('/')+1]

                    print(line("file              : {!s}".format(fn)))

                    if ARES in path:
                        path = path.replace(ARES, '')
                        print(line(("path              : " +\
                            "$ARES{!s}").format(path)))
                    else:
                        print(line("path              : {!s}".format(path)))

    else:
        print(line("Emin (eV)         : {0:.1e}".format(\
            rb.pf['spectrum_Emin'])))
        print(line("Emax (eV)         : {0:.1e}".format(\
            rb.pf['spectrum_Emax'])))

        if rb.pf['spectrum_Emin'] < 13.6:
            if not rb.pf['discrete_lwb']:
                print(line("NOTE              : this is a continuous " +\
                    "radiation field!"))
            else:
                print(line(("NOTE              : discretized over first {} " +\
                    "Ly-n bands").format(rb.pf['lya_nmax'])))
        else:
            print(line("NOTE              : this is a continuous radiation " +\
                "field!"))

    print("#" * width)

    for warning in warnings:
        print_warning(warning)

def print_model_set(mset):
    if rank > 0:
        return

    header = 'Analysis: Model Set'
    print("\n" + ("#" * width))
    print("{0!s} {1!s} {2!s}".format(pre, header.center(twidth), post))
    print("#" * width)

    print(line('-' * twidth))
    print(line('Basic Information'))
    print(line('-' * twidth))

    i = mset.prefix.rfind('/') # forward slash index

    # This means we're sitting in the right directory already
    if i == - 1:
        path = './'
        prefix = mset.prefix
    else:
        path = mset.prefix[0:i+1]
        prefix = mset.prefix[i+1:]

    print(line("path        : {!s}".format(path)))
    print(line("prefix      : {!s}".format(prefix)))
    print(line("N-d         : {}".format(len(mset.parameters))))

    print(line('-' * twidth))
    for i, par in enumerate(mset.parameters):
        print(line("param    #{0!s}: {1!s}".format(str(i).zfill(2), par)))

    print("#" * width)
    print("")
