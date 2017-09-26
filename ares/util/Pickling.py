"""
File: $PERSES/perses/util/Pickling.py
Author: Keith Tauscher
Date: 25 Sep 2017

Description: A wrapper around pickle's reading and writing of objects which
             adds some protections and features, such as checking to make sure
             a file isn't being overwritten and printing when a file is being
             printed.
"""
import os, sys, subprocess, pickle
# uncomment below if you want to import dill
#try:
#    import dill as pickle
#except:
#    import pickle
python_major_version = sys.version_info.major
if python_major_version in [2, 3]:
    is_python3 = (python_major_version == 3)
else:
    raise ValueError("The check for which Python version is being used " +\
        "failed for some unknown reason.")

try:
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    have_mp = True
except:
    size = 1
    rank = 0
    have_mp = False

def write_pickle_file(value, file_name, ndumps=1, open_mode='w',\
    safe_mode=False, verbose=False):
    """
    Writes a pickle file containing the given value at the given file name.
    
    value: the object to pickle
    file_name: the file in which to save the pickled value
    ndumps: number of individual calls to dump.
            If ndumps==1, then value is pickled directly
            If ndumps>1, then value is interpreted as a sequence (list or
                         tuple) of length ndumps of objects to pickle
    open_mode: the opening mode of the pickle file (either 'w' or 'a')
    safe_mode: if True, function is guaranteed not to overwrite existing files
    verbose: Boolean determining whether a string should be printed describing
             what file is being saved by this function
    """
    if os.path.exists(file_name) and safe_mode:
        raise NotImplementedError("A pickle file is being created in place " +\
                                  "of an existing file. If this is what " +\
                                  "you want, set safe_mode=False in the " +\
                                  "keyword arguments of your call to " +\
                                  "write_pickle_file.")
    if open_mode not in ['a', 'w']:
        raise ValueError("The mode with which to open the pickle file for " +\
            "writing was not understood. It should be either 'a' (for " +\
            "append -- existing data won't be deleted) or 'w' (for write " +\
            "-- existing data will be deleted).")
    if ndumps == 1:
        objects_to_pickle = [value]
    else:
        objects_to_pickle = value
    try:
        with open(file_name, open_mode + 'b') as pickle_file:
            if verbose and (rank == 0):
                print("Writing {!s}...".format(file_name))
            for object_to_pickle in objects_to_pickle:
                if is_python3:
                    # extra 'protocol' keyword argument necessary for python2
                    # to be able to unpickle python3-pickled files
                    pickle.dump(object_to_pickle, pickle_file, protocol=2)
                else:
                    pickle.dump(object_to_pickle, pickle_file, -1)
    except IOError as err:
        if err.errno == 2:
            raise IOError("A write of a pickle file (" + file_name + ") " +\
                          "was attempted in a directory which does not exist.")
        elif err.errno == 13:
            raise IOError("A write of a pickle file (" + file_name + ") " +\
                          "was attempted in a directory (or, less likely, " +\
                          "a file) to which the user does not have write " +\
                          "permissions.")
        else:
            raise err

def read_pickle_file(file_name, nloads=1, verbose=False):
    """
    Reads the pickle file located at the given file name.
    
    file_name: the name of the file which is assumed to be a pickled object
    nloads: number of separate calls to load to make.
            If nloads==1, one object is loaded and returned
            If nloads>1, a sequence of nloads objects is loaded and returned
            If nloads is None, load is called until it raises an error and this
                               function returns a sequence whose length is
                               given by the number of successful calls to
                               pickle.load and whose values are the unpickled
                               objects
    verbose: Boolean determining whether string should be printed echoing the
             location of the file being read
    
    returns: the object(s) pickled in the file located at file_name
    """
    try:
        with open(file_name, 'rb') as pickle_file:
            if verbose and (rank == 0):
                print("Reading {!s}...".format(file_name))
            return_val = []
            if nloads is not None:
                iload = 0
            while True:
                if (nloads is not None) and (iload == nloads):
                    break
                try:
                    if is_python3:
                        return_val.append(\
                            pickle.load(pickle_file, encoding='latin1'))
                    else:
                        return_val.append(pickle.load(pickle_file))
                except EOFError:
                    if nloads is None:
                        break
                    else:
                        raise
                if nloads is not None:
                    iload += 1
            if (nloads is not None) and (len(return_val) == 1):
                return_val = return_val[0]
            return return_val
    except IOError as err:
        if err.errno == 2:
            raise IOError(("A pickle file ({!s}) which does not exist was " +\
                          "attempted to be read.").format(file_name))
        elif err.errno == 13:
            raise IOError(("A pickle file ({!s}) could not be read because " +\
                          "the user does not have read permissions.").format(\
                          file_name))
        else:
            raise err


def delete_file_if_clobber(file_name, clobber=False, verbose=True):
    """
    Deletes the given file if it exists and clobber is set to True.
    
    file_name: the file to delete, if it exists
    clobber: if clobber is False, nothing is done
    verbose: if verbose is True, a message indicating the file is being removed
             is printed to the user
    """
    if clobber and os.path.exists(file_name):
        if verbose:
            print("Removing {!s}...".format(file_name))
        subprocess.call(['rm', file_name])


def delete_file(file_name, verbose=True):
    """
    Deletes the given file if it exists.
    
    file_name: the file to delete, if it exists
    verbose: if verbose is True, a message indicating the file is being removed
             is printed to the user
    """
    delete_file_if_clobber(file_name, clobber=True, verbose=verbose)

def overwrite_pickle_file(quantity, file_name, verbose=True):
    """
    Writes the given quantity to a pickle file at the given path, regardless of
    whether there is a file currently existing there.
    
    quantity: the object to save in the pickle file at the given path
    file_name: path to name of desired file. If a file already exists there, it
               will be deleted.
    verbose: if verbose is True, a message indicating the file is being removed
             is printed to the user
    """
    delete_file(file_name, verbose=verbose)
    write_pickle_file(quantity, file_name, safe_mode=False, verbose=verbose)

