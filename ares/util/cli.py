"""
A module for downloading remote data.

Author: Jordan Mirocha and Paul La Plante
Affiliation: JPL
Created on: Sun Feb 5 12:51:32 PST 2023
"""
import argparse
import os
import re
import sys
import tarfile
import zipfile
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import numpy as np
import h5py

from . import ParameterBundle
from .. import __version__
from ..data import ARES
from ..physics import HaloMassFunction
from ..populations import GalaxyPopulation
from ..solvers import OpticalDepth

# define helper function
def read_FJS10():
    E_th = [13.6, 24.6, 54.4]

    # fmt: off
    # Ionized fraction points and corresponding files
    x = np.array(
        [
            1.0e-4, 2.318e-4, 4.677e-4, 1.0e-3, 2.318e-3,
            4.677e-3, 1.0e-2, 2.318e-2, 4.677e-2, 1.0e-1,
            0.5, 0.9, 0.99, 0.999,
        ]
    )

    xi_files = [
        "xi_0.999.dat", "xi_0.990.dat", "xi_0.900.dat", "xi_0.500.dat",
        "log_xi_-1.0.dat", "log_xi_-1.3.dat", "log_xi_-1.6.dat",
        "log_xi_-2.0.dat", "log_xi_-2.3.dat", "log_xi_-2.6.dat",
        "log_xi_-3.0.dat", "log_xi_-3.3.dat", "log_xi_-3.6.dat",
        "log_xi_-4.0.dat"
    ]
    # fmt: on

    xi_files.reverse()

    # Make some blank arrays
    energies = np.zeros(258)
    heat = np.zeros([len(xi_files), 258])
    fion = np.zeros_like(heat)
    fexc = np.zeros_like(heat)
    fLya = np.zeros_like(heat)
    fHI = np.zeros_like(heat)
    fHeI = np.zeros_like(heat)
    fHeII = np.zeros_like(heat)

    # Read in energy and fractional heat deposition for each ionized fraction.
    for i, fn in enumerate(xi_files):
        # Read data
        nrg, f_ion, f_heat, f_exc, n_Lya, n_ionHI, n_ionHeI, n_ionHeII, \
            shull_heat = np.loadtxt("x_int_tables/{!s}".format(fn), skiprows=3,
                unpack=True)

        if i == 0:
            for j, energy in enumerate(nrg):
                energies[j] = energy

        for j, h in enumerate(f_heat):
            heat[i][j] = h
            fion[i][j] = f_ion[j]
            fexc[i][j] = f_exc[j]
            fLya[i][j] = (n_Lya[j] * 10.2) / energies[j]
            fHI[i][j] = (n_ionHI[j] * E_th[0]) / energies[j]
            fHeI[i][j] = (n_ionHeI[j] * E_th[1]) / energies[j]
            fHeII[i][j] = (n_ionHeII[j] * E_th[2]) / energies[j]

    # We also want the heating as a function of ionized fraction for each photon energy.
    heat_xi = np.array(list(zip(*heat)))
    fion_xi = np.array(list(zip(*fion)))
    fexc_xi = np.array(list(zip(*fexc)))
    fLya_xi = np.array(list(zip(*fLya)))
    fHI_xi = np.array(list(zip(*fHI)))
    fHeI_xi = np.array(list(zip(*fHeI)))
    fHeII_xi = np.array(list(zip(*fHeII)))

    # Write to hfd5
    with h5py.File("secondary_electron_data.hdf5", "w") as h5f:
        h5f.create_dataset("electron_energy", data=energies)
        h5f.create_dataset("ionized_fraction", data=np.array(x))
        h5f.create_dataset("f_heat", data=heat_xi)
        h5f.create_dataset("fion_HI", data=fHI_xi)
        h5f.create_dataset("fion_HeI", data=fHeI_xi)
        h5f.create_dataset("fion_HeII", data=fHeII_xi)
        h5f.create_dataset("f_Lya", data=fLya_xi)
        h5f.create_dataset("fion", data=fion_xi)
        h5f.create_dataset("fexc", data=fexc_xi)

    return

# define data sources
_bpass_v1_links = [
    f"sed_bpass_z{zval}_tar.gz" for zval in ["001", "004", "008", "020", "040"]
]
_bpass_tests = "https://www.dropbox.com/s/bq10l5f6gzqqvu7/sed_degraded.tar.gz?dl=1"

# Auxiliary data downloads
# Format: [URL, file1, file2, ..., file to run when done]
aux_data = {
    "halos": [
        "https://www.dropbox.com/s/8df7rsskr616lx5/halos.tar.gz?dl=1",
        "halos.tar.gz",
        None,
    ],
    "inits": [
        "https://www.dropbox.com/s/c6kwge10c8ibtqn/inits.tar.gz?dl=1",
        "inits.tar.gz",
        None,
    ],
    "optical_depth": [
        "https://www.dropbox.com/s/ol6240qzm4w7t7d/tau.tar.gz?dl=1",
        "tau.tar.gz",
        None,
    ],
    "secondary_electrons": [
        "https://www.dropbox.com/s/jidsccfnhizm7q2/elec_interp.tar.gz?dl=1",
        "elec_interp.tar.gz",
        read_FJS10,
    ],
    "starburst99": [
        "http://www.stsci.edu/science/starburst99/data", "data.tar.gz", None
    ],
    "bpass_v1": [
        "http://bpass.auckland.ac.nz/2/files"
    ] + _bpass_v1_links + [_bpass_tests] + [None],
    "bpass_v1_stars": [
        "http://bpass.auckland.ac.nz/1/files", "starsmodels_tar.gz", None
    ],
    "umachine-data": [
        "http://halos.as.arizona.edu/UniverseMachine/DR1",
        "umachine-dr1-obs-only.tar.gz",
        None,
    ],
    "edges": [
        "http://loco.lab.asu.edu/download",
        "790/figure1_plotdata.csv",
        "792/figure2_plotdata.csv",
        None,
    ],
    "nircam": [
        "https://jwst-docs.stsci.edu/files/97978094/97978135/1/1596073152953",
        "nircam_throughputs_22April2016_v4.tar.gz",
        None,
    ],
    "wfc3": [
        "https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation"
        "/wfc3/performance/throughputs/_documents/",
        "IR.zip",
        None,
    ],
    "wfc": [
        "https://www.dropbox.com/s/zv8qomgka9fkiek/wfc.tar.gz?dl=1",
        "wfc.tar.gz",
        None,
    ],
    "irac": [
        "https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles"
        "/spectralresponse/",
        "080924ch1trans_full.txt",
        "080924ch2trans_full.txt",
        None,
    ],
    "roman": [
        "https://roman.gsfc.nasa.gov/science/RRI/",
        "Roman_effarea_20201130.xlsx",
        None,
    ],
    "rubin": [
        "https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data",
        "throughputs_aug_2021.tgz",
        None,
    ],
    "spherex": [
        "https://github.com/SPHEREx/Public-products/archive/refs/heads",
        "master.zip",
        None,
    ],
    "wise": [
        "https://www.astro.ucla.edu/~wright/WISE",
        "RSR-W1.txt",
        "RSR-W2.txt",
        None,
    ],
    "planck": [
        "https://pla.esac.esa.int/pla/aio",
        "product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE"
        "-lowl-lowE_R3.00.zip",
        "product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-zre6p5_R3.01.zip",
        "product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM_R3.01.zip",
        None,
    ],
}

# define which files are needed for which things
datasets = {
    "extra": [
        "nircam",
        "irac",
        "roman",
        "edges",
        "bpass_v1_stars",
    ],
    "tests": [
        "inits",
        "secondary_electrons",
        "halos",
        "wfc",
        "wfc3",
        "planck",
        "bpass_v1",
        "optical_depth",
    ],
    "test_files": [
        "inits.tar.gz",
        "elec_interp.tar.gz",
        "halos.tar.gz",
        "IR.zip",
        "wfc.tar.gz",
        aux_data["planck"][1],
        "sed_degraded.tar.tz",
        "tau.tar.gz",
    ],
}

def generate_optical_depth_tables(path, **kwargs):
    """
    Generate optical depth tables for ARES.

    Parameters
    ----------
    path : str
        The full path to where to save output files.
    kwargs
        Keyword arguments that will be passed to the ares.solvers.OpticalDepth
        instance.

    Returns
    -------
    None
    """
    # go to path
    os.chdir(path)

    # initialize radiation background
    def_kwargs = {
        "tau_Emin": 2e2,
        "tau_Emax": 3e4,
        "tau_Emin_pin": True,
        "tau_fmt": "hdf5",
        "tau_redshift_bins": 400,
        "approx_He": 1,
        "include_He": 1,
        "initial_redshift": 60,
        "final_redshift": 5,
        "first_light_redshift": 60,
    }

    # update defaults with kwargs
    def_kwargs.update(kwargs)

    # Create OpticalDepth instance
    igm = OpticalDepth(**def_kwargs)

    # Impose an ionization history: neutral for all times
    igm.ionization_history = lambda z: 0.0

    # Tabulate tau and save
    tau = igm.TabulateOpticalDepth()
    igm.save(suffix=kwargs["tau_fmt"], clobber=False)
    return

def make_tau(path):
    """
    Generate a whole bunch of optical depth files.

    This function replicates a lot of the old funcitonality of the pack_tau.sh
    shell script.

    Parameters
    ----------
    path : str
        The full path to the directory to save output tables to.

    Returns
    -------
    None
    """
    generate_optical_depth_tables(path, tau_redshift_bins=400, include_He=1)
    generate_optical_depth_tables(path, tau_redshift_bins=400, include_He=0)
    generate_optical_depth_tables(path, tau_redshift_bins=1000, include_He=1)
    generate_optical_depth_tables(path, tau_redshift_bins=1000, include_He=0)

    return

def generate_hmf_tables(path, **kwargs):
    """
    Generate halo mass function tables for ARES.

    Parameters
    ----------
    path : str
        The full path for where to save output files.
    kwargs
        Keyword arguments passed to the ares.physics.HaloMassFunction object.

    Returns
    -------
    None
    """
    # go to path
    os.chdir(path)

    # initialize hmf values
    def_kwargs = {
        "halo_mf": "Tinker10",
        "halo_logMmin": 4,
        "halo_logMmax": 18,
        "halo_dlogM": 0.01,

        "halo_fmt": "hdf5",
        "halo_table": None,
        "halo_wdm_mass": None,

        # Can do constant timestep instead of constant dz
        "halo_dt": 10,
        "halo_tmin": 30.0,
        "halo_tmax": 13.7e3,  # Myr

        # Cosmology
        "cosmology_id": "best",
        "cosmology_name": "planck_TTTEEE_lowl_lowE",
    }

    def_kwargs.update(kwargs)

    halos = HaloMassFunction(halo_mf_analytic=False, halo_mf_load=False, **kwargs)
    halos.info

    try:
        halos.save_hmf(fmt="hdf5", clobber=False)
    except IOError as err:
        print(err)
    return

def generate_halo_histories(path, fn_hmf):
    """
    Generate halo histories.

    Parameters
    ----------
    path : str
        The full path to the directory to save output files to.
    fn_hmf : str
        The name of the output file to produce.

    Returns
    -------
    None
    """
    # go to path
    os.chdir(path)

    # define parameters
    pars = (
        ParameterBundle("mirocha2017:base").pars_by_pop(0, 1)
        + ParameterBundle("mirocha2017:dflex").pars_by_pop(0, 1)
    )

    pars["halo_mf_table"] = fn_hmf

    with h5py.File(fn_hmf, "r") as h5f:
        grp = h5f["cosmology"]

        cosmo_pars = {}
        cosmo_pars["cosmology_name"] = grp.attrs.get("cosmology_name")
        cosmo_pars["cosmology_id"] = grp.attrs.get("cosmology_id")

        for key in grp:
            buff = np.zeros(1)
            grp[key].read_direct(buff)
            cosmo_pars[key] = buff[0]

        print(f"Read cosmology from {fn_hmf}")

    pars.update(cosmo_pars)

    # We might periodically tinker with these things but these are good defaults.
    pars["pop_Tmin"] = None
    pars["pop_Mmin"] = 1e4
    pars["halo_hist_dlogM"] = 0.1  # Mass bins [in units of Mmin]
    pars["halo_hist_Mmax"] = 10  # by default, None, but 10 is good enough for most apps

    pop = GalaxyPopulation(**pars)

    if "npz" in fn_hmf:
        pref = fn_hmf.replace(".npz", "").replace("halo_mf", "halo_hist")
    elif "hdf5" in fn_hmf:
        pref = fn_hmf.replace(".hdf5", "").replace("halo_mf", "halo_hist")
    else:
        raise IOError("Unrecognized file format for HMF ({})".format(fn_hmf))

    if pars["halo_hist_Mmax"] is not None:
        pref += "_xM_{:.0f}_{:.2f}".format(pars["halo_hist_Mmax"], pars["halo_hist_dlogM"])

    fn = "{}.hdf5".format(pref)
    if not os.path.exists(fn):
        print("Running new trajectories...")
        zall, hist = pop.Trajectories()

        with h5py.File(fn, "w") as h5f:
            # Save halo trajectories
            for key in hist:
                if key not in ["z", "t", "nh", "Mh", "MAR"]:
                    continue
                h5f.create_dataset(key, data=hist[key])

        print("Wrote {}".format(fn))
    else:
        print("File {} exists. Exiting.".format(fn))
    return

def make_halos(path):
    """
    Generate a whole bunch of halo mass function tables.

    This function replicates a lot of the old funcitonality of the pack_hmf.sh
    shell script.

    Parameters
    ----------
    path : str
        The full path to the directory to save output files to.

    Returns
    -------
    None
    """
    generate_hmf_tables(path, halo_mf="ST")
    generate_hmf_tables(path, halo_mf="PS", halo_zmin=5, halo_zmax=30, halo_dz=1)
    generate_hmf_tables(path, halo_mf="ST", halo_dt=1, halo_tmin=30, halo_tmax=1000)
    generate_halo_histories(
        path,
        "halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5",
    )
    return

def make_data_dir(path=ARES):
    """
    Make a data directory at the specified path.

    Parameters
    ----------
    path : str, optional
        The path to the directory to make. The directory will only be made if it
        does not yet exist. Defaults to the ARES directory defined in the
        package.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.mkdir(path)

    return

def clean_files(args):
    """
    Clean up downloaded ARES files.

    Parameters
    ----------
    args : ArgumentParser instance
        An ArgumentParser object that contains the arguments from the command line.

    Returns
    -------
    None
    """
    # get list of datasets
    available_dsets = [key.lower() for key in aux_data.keys()]

    # figure out what to delete
    if args.dataset.lower() == "all":
        dsets = available_dsets
    elif args.dataset.lower() not in available_dsets:
        raise ValueError(
            f"dataset {args.dataset} is not available. Possible options are: "
            f"{available_dsets}"
        )
    else:
        dsets = [args.dataset.lower()]

    # echo out what we would delete
    if args.dry_run:
        for dset in dsets:
            full_path = os.path.join(args.path, dset, aux_dsets[dset][1])
            print(f"Running in dry-run mode; would remove {full_path}")
    else:
        for dset in dsets:
            full_path = os.path.join(args.path, dset, aux_dsets[dset][1])
            print(f"Removing {full_path}...")
            os.remove(full_path)

    return

def _do_download(full_path, dl_link):
    try:
        urlretrieve(dl_link, full_path)
    except (URLError, HTTPError) as error:
        print(f"Error downloading file {dl_link} to {full_path}")
        print(f"error: {error}")
    return

def download_files(args):
    """
    Download auxiliary data files for ARES.

    Parameters
    ----------
    args : ArgumentParser instance
        An ArgumentParser object that contains the arguments from the command line

    Returns
    -------
    None
    """
    # get list of datasets
    available_dsets = [key.lower() for key in aux_data.keys()]

    # figure out what to download
    if args.dataset.lower() == "all":
        dsets = available_dsets
    elif args.dataset.lower() not in available_dsets:
        raise ValueError(
            f"dataset {args.dataset} is not available. Possible options are: "
            f"{available_dsets}"
        )
    else:
        dsets = [args.dataset.lower()]

    # check to see if data exists
    if args.dry_run:
        for dset in dsets:
            full_path = os.path.join(args.path, dset, aux_data[dset][1])
            if os.path.exists(full_path):
                if args.fresh:
                    print(f"Running in dry-run mode; would re-download {full_path}")
                else:
                    print(
                        f"{full_path} already exists; rerun with --fresh to "
                        "force download"
                    )
            else:
                print("Running in dry-run mode; would download {full_path}")
    else:
        for dset in dsets:
            parent_dir = os.path.join(args.path, dset)
            full_path = os.path.join(parent_dir, aux_data[dset][1])
            if os.path.exists(full_path):
                if args.fresh:
                    _do_download(full_path, dl_link)
                else:
                    print(
                        f"{full_path} already exists; rerun with --fresh to "
                        "force download"
                    )
            else:
                make_data_dir(parent_dir)
                _do_download(full_path, dl_link)

        if aux_data[dset][2] is not None:
            # this is a callable with no arguments
            aux_data[dset][2]()

    return

def generate_data(args):
    """
    Generate auxiliary data files for ARES.

    Parameters
    ----------
    args : ArgumentParser instance
        An ArgumentParser object that contains the arguments from the command line

    Returns
    -------
    None
    """
    # figure out what to generate
    available_dsets = ["tau", "hmf"]
    if args.dataset.lower() == "all":
        dsets = available_dsets
    elif args.dataset.lower() not in available_dsets:
        raise ValueError(
            f"dataset {args.dataset} is not available. Possible options are: "
            f"{available_dsets}"
        )
    else:
        dsets = [args.dataset.lower()]

    if args.dry_run:
        for dset in dsets:
            if dset == "tau":
                print("Running in dry-run mode; would generate optical depth data")
            elif dset == "hmf":
                print("Running in dry-run mode; would generate halo mass function data")
    else:
        for dset in dsets:
            if dset == "tau":
                make_tau(args.path)
            elif dset == "hmf":
                make_halos(args.path)

    return

def config_clean_subparser(subparser):
    """
    Add the subparser for the "clean" sub-command.

    Parameters
    ----------
    subparser : ArgumentParser subparser object
        The subparser object to add sub-command options to.

    Returns
    -------
    None
    """
    doc = """
    Clean up remote files for ARES.
    """
    hlp = "clean up files for ARES"
    sp = subparser.add_parser(
        "clean",
        description=doc,
        help=hlp,
    )
    sp.add_argument(
        "dataset",
        metavar="DATASET",
        type=str,
        help="dataset to remove",
        default="all",
    )
    sp.set_defaults(func=clean_files)

    return

def config_download_subparser(subparser):
    """
    Add the subparser for the "download" sub-command.

    Parameters
    ----------
    subparser : ArgumentParser subparser object
        The subparser object to add sub-command options to.

    Returns
    -------
    None
    """
    doc = """
    Download remote files for ARES.
    """
    hlp = "download files for ARES"
    sp = subparser.add_parser(
        "download",
        description=doc,
        help=hlp,
    )
    sp.add_argument(
        "dataset",
        metavar="DATASET",
        type=str,
        help="dataset to download",
        default="all",
    )
    sp.add_argument(
        "--fresh",
        help="whether to force a new download or not",
        action="store_true",
    )
    sp.set_defaults(func=download_files)

    return

def config_generate_subparser(subparser):
    """
    Add the subparser for the "generate" sub-command.

    Parameters
    ----------
    subparser : ArgumentParser subparser object
        The subparser object to add sub-command options to.

    Returns
    -------
    None
    """
    doc = """
    Generate lookup files for ARES.
    """
    hlp = "generate files for ARES"
    sp = subparser.add_parser(
        "generate",
        description=doc,
        help=hlp,
    )
    sp.add_argument(
        "dataset",
        metavar="DATASET",
        type=str,
        help="dataset to generate",
        default="all",
    )
    sp.add_argument(
        "--fresh",
        help="whether to force a new generation or not",
        action="store_true",
    )
    sp.set_defaults(func=generate_data)

    return

# make the base parser
def generate_parser():
    """
    Make an ares argparser.

    The `ap` object returned contains subparsers for all sub-commands.

    Parameters
    ----------
    None

    Returns
    -------
    ap : ArgumentParser object
        The populated ArgumentParser with subcommands.
    """
    ap = argparse.ArgumentParser(
        description="remote.py is a command for downloading remote datasets for ARES"
    )
    ap.add_argument(
        "-V",
        "--version",
        action="version",
        version="ares {}".format(__version__),
        help="Show ares version and exit",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what actions would be taken without doing anything",
    )
    ap.add_argument(
        "-p",
        "--path",
        default=ARES,
        help="path to download files to. Defaults to ~/.ares",
    )

    # add subparsers
    sub_parsers = ap.add_subparsers(metavar="command", dest="cmd")
    config_clean_subparser(sub_parsers)
    config_download_subparser(sub_parsers)
    config_generate_subparser(sub_parsers)

    return ap

# target function for cli invocation
def main():
    # make a parser and run the specified command
    parser = generate_parser()
    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)

    return

if __name__ == "__main__":
    sys.exit(main())
