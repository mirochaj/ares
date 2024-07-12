"""
Bruzual & Charlot 2003
"""

import numpy as np
from ares.data import ARES
from ares.physics.Constants import Lsun

_input = ARES + '/bc03/'

metallicities = \
{
 'm72': 0.05,
 'm62': 0.02,
 'm52': 0.008,
 'm42': 0.004,
 'm32': 0.0004,
 'm22': 0.0001,
}

mvals = metallicities.values()

def _kwargs_to_fn(**kwargs):
    """
    Determine filename of appropriate BPASS lookup table based on kwargs.
    """

    #assert 'source_tracks' in kwargs

    path = 'models/Padova1994'

    assert kwargs['source_ssp']


    #assert kwargs['source_imf'] == 'chabrier'

    path += '/chabrier/'

    # All files share this prefix
    fn = 'bc2003_hr'

    Z = kwargs['source_Z']
    iZ = list(mvals).index(Z)
    key = list(metallicities.keys())[iZ]
    fn += f'_{key}_chab_ssp.ised_ASCII'

    if kwargs['source_sed_degrade'] is not None:
        fn += '.deg{}'.format(kwargs['source_sed_degrade'])

    return _input + path + fn

def _load(**kwargs):
    fn = _kwargs_to_fn(**kwargs)

    with open(fn, 'r') as f:
        times = []
        waves = []
        ct = 0
        ct2 = 0

        spec = None
        dunno = None
        # 221 times, 6900 wavelengths
        data = np.zeros((6900, 221))
        for i in range(332582):
            line = f.readline().split()

            if i == 0:
                times.extend([float(element) for element in line[1:]])
            elif i < 37:
                times.extend([float(element) for element in line])
            elif i in [37, 38, 39, 40, 41]:
                continue
            elif i == 42:
                waves.extend([float(element) for element in line[1:]])
            elif i < 679:
                waves.extend([float(element) for element in line])
            else:
                if float(line[0]) == 6900:
                    if ct > 0:
                        # This shouldn't be necessary. What are the extra
                        # 53 elements?
                        data[:,ct-1] = np.array(spec)[0:6900]

                    ct += 1

                    spec = [float(element) for element in line[1:]]

                else:
                    spec.extend([float(element) for element in line])


            #if ct == 3:
            #    break

        f.close()

        assert ct == 221

    # Done. Convert times to Myr, SEDs to erg/s, and return
    return np.array(waves), np.array(times)[1:] / 1e6, data[:,1:] * Lsun * 1e6, fn
