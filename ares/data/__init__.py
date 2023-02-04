HOME = os.getenv("HOME")
ARES = f"{HOME}/.ares"

# check that directory exists
if not os.path.exists(ARES):
    raise IOError("The directory ~/.ares does not exist. Please make it, or re-run package installation.")
