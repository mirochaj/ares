import os
import importlib
import numpy as np

HOME = os.getenv("HOME")
ARES = f"{HOME}/.ares"

# check that directory exists
if os.path.islink(ARES):
    pass
elif not os.path.exists(ARES):
    raise IOError(f"The directory {ARES} does not exist. Please make it, or re-run package installation.")

class DummyDataset(object):
    def __init__(self):
        pass

def fix_cosmology(data, cosmo_ours=None):
    """
    This routine goes through a given dataset and converts its 
    cosmology to ours.
    """

    if cosmo_ours is None:
        print(f"! Must provide `cosmo_ours` to correct cosmological inconsistencies!")
        return data

    from ..physics import Cosmology

    if isinstance(cosmo_ours, Cosmology):
        cosm_ours = cosmo_ours
    else:
        cosm_ours = Cosmology(**cosmo_ours)

    cosm_theirs = Cosmology(cosmology_name='user',
        **data.cosmo)
        
    new_data = DummyDataset()
    # A few things need to get passed along
    new_data.redshifts = data.redshifts
    new_data.units = data.units
    if hasattr(data, 'zbins'):
        new_data.zbins = data.zbins
    # Stores everything
    new_data.data = {}
    # Loop through contents of data and apply correction
    for element in data.data.keys():
        # `element` will be 'lf' or 'smf' etc.
        new_data.data[element] = {}
        # The next level of sorting is by redshift
        for red in data.data[element].keys():
            new_data.data[element][red] = {}
            z = np.mean(red) if type(red) == tuple else red
            # Volume
            Vcorr = (cosm_ours.get_hubble(z) \
                   / cosm_theirs.get_hubble(z))**3
            # Lum
            # SFR
            # mass
            Mcorr = (cosm_ours.get_hubble(z) \
                   / cosm_theirs.get_hubble(z))**-2
            
            # Magnitude
            magcorr = 5 * np.log(cosm_theirs.get_hubble(z)) \
                    - 5 * np.log(cosm_ours.get_hubble(z))
            # Is this ever not another dictionary?

            for key in data.data[element][red]:
                
                ydat = np.array(data.data[element][red][key])
                # Correct volume
                if key == 'phi':
                    if data.units[key].startswith('log10'):
                        new_data.data[element][red][key] = \
                            np.log10(Vcorr * 10**ydat)
                    else:
                        new_data.data[element][red][key] = Vcorr * ydat
                elif key == 'mass':
                    if data.units[key].startswith('log10'):
                        new_data.data[element][red][key] = \
                            np.log10(Mcorr * 10**ydat)
                    else:
                        new_data.data[element][red][key] = Mcorr * ydat
                elif key == 'M':
                    if data.units[key] == 'mags_abs':
                        new_data.data[element][red][key] = magcorr + ydat
                    else:
                        new_data.data[element][red][key] = ydat
                else:
                    # Errors...what else?
                    new_data.data[element][red][key] = ydat
    
    return new_data

def read(prefix, path=None, verbose=True, cosmo_ours=None):
    """
    Read data from the literature.

    Parameters
    ----------
    prefix : str
        Everything preceeding the '.py' in the name of the module.
    path : str
        If you want to look somewhere besides $ARES/input/litdata, provide
        that path here.

    """

    # First: try to import from ares.data (i.e., right here)
    mod = importlib.import_module(f'ares.data.{prefix}')
    if mod is not None:
        pass
    else:
        if path is not None:
            loc = path
        else:
            fn = f"{prefix}.py"
            has_local = os.path.exists(os.path.join(os.getcwd(), fn))
            has_home = os.path.exists(os.path.join(HOME, ".ares", fn))
    
            # Load custom defaults
            if has_local:
                loc = os.getcwd()
            elif has_home:
                loc = os.path.join(HOME, ".ares")
            else:
                return None
    
            if has_local + has_home > 1:
                print("WARNING: multiple copies of {!s} found.".format(prefix))
                print("       : precedence: CWD -> $HOME -> $ARES/input/litdata")
    
        mod = importlib.__import__(f"{loc}/{prefix}")
    
        # Save this for sanity checks later
        mod.path = loc

    if (cosmo_ours is None) or (mod.cosmo is None):
        return mod
    else:
        return fix_cosmology(mod, cosmo_ours=cosmo_ours)
