import os
import importlib

HOME = os.getenv("HOME")
ARES = f"{HOME}/.ares"

# check that directory exists
if not os.path.exists(ARES):
    raise IOError("The directory ~/.ares does not exist. Please make it, or re-run package installation.")

def read(prefix, path=None, verbose=True):
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
        return mod

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

    return mod
