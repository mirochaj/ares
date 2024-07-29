"""
Bruzual & Charlot 2003 (2013 UPDATE)

Help from https://github.com/cmancone/easyGalaxy/blob/master/ezgal/utils.py
on how to read *.ised files. Thanks!
"""

import os
import sys
import array
import pickle
import numpy as np
from ares.data import ARES
from ares.physics.Constants import Lsun

_input = ARES + '/bc03_2013/'

metallicities_p94 = \
{
 'm72': 0.05,
 'm62': 0.02,
 'm52': 0.008,
 'm42': 0.004,
 'm32': 0.0004,
 'm22': 0.0001,
}

metallicities_p00 = \
{
 'm172': 0.05,
 'm162': 0.02,
 'm152': 0.008,
 'm142': 0.004,
 'm132': 0.0004,
 'm122': 0.0001,
}

metallicities = metallicities_p00

def _kwargs_to_fn(**kwargs):
    """
    Determine filename of appropriate BPASS lookup table based on kwargs.
    """

    #assert 'source_tracks' in kwargs

    path = f"bc03/{kwargs['source_tracks']}"
    if kwargs['source_tracks'] == 'Padova1994':
        metallicities = metallicities_p94
    elif kwargs['source_tracks'] == 'Padova2000':
        metallicities = metallicities_p00
    else:
        assert kwargs['source_tracks'] == 'Geneva1994', \
            "Only know source_tracks in [Padova1994,Padova2000,Geneva1994]"
        assert kwargs['source_Z'] == 0.02, \
            "For Geneva 1994 tracks, BC03 only contains solar metallicity"

        metallicities = {'m64': 0.02}

    mvals = metallicities.values()

    #assert kwargs['source_imf'] == 'chabrier'

    path += f"/{kwargs['source_imf']}/"

    # All files share this prefix
    fn = 'bc2003_hr_stelib'

    Z = kwargs['source_Z']
    iZ = list(mvals).index(Z)
    key = list(metallicities.keys())[iZ]
    fn += f"_{key}_{kwargs['source_imf'][0:4]}_ssp.ised"

    if not kwargs['source_ssp']:
        fn += '_csfh'

    if kwargs['source_sed_degrade'] is not None:
        fn += '.deg{}'.format(kwargs['source_sed_degrade'])

    return _input + path + fn

def _read_binary(fhandle, type='i', number=1, swap=False):
    '''
    res = ezgal.utils._read_binary( fhandle, type='i', number=1, swap=False )
    reads 'number' binary characters of type 'type' from file handle 'fhandle'
    returns the value (for one character read) or a numpy array.
    set swap=True to byte swap the array after reading
    '''

    if (sys.version_info >= (3, 0)) & (type == 'c'):
       ##  unsigned char in python 2.
       ##  https://docs.python.org/2/library/array.html
       ##  https://docs.python.org/3/library/array.html
       ##  type  =  'B'  ##  unsigned char in python 3.
       ##  type  =  'b'  ##  signed char in python 3.

       import warnings
       type = 'B'
       warnings.warn('Reassigning unsigned char type (c to B) as per python 3.')

    arr = array.array(type)
    arr.fromfile(fhandle, number)

    if swap:
        arr.byteswap()

    if len(arr) == 1:
        return arr[0]

    else:
        return np.asarray(arr)


def _read_ised(file):
    """ ( seds, ages, vs ) = ezgal.utils.read_ised( file )

	Read a bruzual and charlot binary ised file.

	:param file: The name of the ised file
	:type file: string
	:returns: A tuple containing model data
	:rtype: tuple

	.. note::
		All returned variables are numpy arrays. ages and vs are one
        dimensional arrays, and seds has a shape of (vs.size, ages.size)

	**units**
	Returns units of:

	=============== ===============
	Return Variable   Units
	=============== ===============
	seds            Ergs/s/cm**2/Hz
	ages            Years
	vs              Hz
	=============== ===============
	"""

    if not (os.path.isfile(file)):
        raise ValueError(f"The specified model file was not found! {file}")

    print('Reading .ised:  %s' % str(file))

    # open the ised file
    fh = open(file, 'rb')

    # start reading
    junk = _read_binary(fh)
    nages = _read_binary(fh)

    print('hi', nages)

    # first consistency check
    if nages < 1 or nages > 2000:
        raise ValueError(
            'Problem reading ised file - unexpected data found for the number of ages!')

    # read ages
    ages = np.asarray(_read_binary(fh, type='f', number=nages))

    # read in a bunch of stuff that I'm not interested in but which I read like
    # this to make sure I get to the right spot in the file
    junk = _read_binary(fh, number=2)
    iseg = _read_binary(fh, number=1)

    if iseg > 0:
        junk = _read_binary(fh, type='f', number=6 * iseg)

    junk = _read_binary(fh, type='f', number=3)
    junk = _read_binary(fh)
    junk = _read_binary(fh, type='f')
    junk = _read_binary(fh, type='c', number=80)
    junk = _read_binary(fh, type='f', number=4)
    junk = _read_binary(fh, type='c', number=160)
    junk = _read_binary(fh)
    junk = _read_binary(fh, number=3)

    # read in the wavelength data
    nvs = _read_binary(fh)

    # consistency check
    if nvs < 10 or nvs > 12000:
        raise ValueError('Problem reading ised file - unexpected data found for the number of wavelengths!')

    # read wavelengths and convert to frequency (comes in as Angstroms)
    # also reverse the array so it will be sorted after converting to frequency
    ls = _read_binary(fh, type='f', number=nvs)[::-1]

    # create an array for storing SED info
    seds = np.zeros((nvs, nages))

    # now loop through and read in all the ages
    for i in range(nages):
        junk = _read_binary(fh, number=2)
        nv = _read_binary(fh)
        if nv != nvs:
            raise ValueError(
                'Problem reading ised file - unexpected data found while reading seds!')

        seds[:, i] = _read_binary(fh, type='f', number=nvs)[::-1]
        nx = _read_binary(fh)
        junk = _read_binary(fh, type='f', number=nx)

    # now convert the seds from Lo/A to ergs/s/Hz
    seds *= Lsun * 1e6

    fh.close()

    return ls[-1::-1], ages[1:] / 1e6, seds[-1::-1,1:], file

def _load(**kwargs):
    fn = _kwargs_to_fn(**kwargs)

    # Simpler! We made this.
    if fn.endswith('csfh'):
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        return data['waves'], data['t'], data['data'], fn

    return _read_ised(fn)
