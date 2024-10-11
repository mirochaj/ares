"""

tests/test_obs_survey.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Tue 21 Dec 2021 14:38:11 EST

Description:

"""

import numpy as np
from ares.obs import Survey, get_filters_from_waves

def test():
    hst = Survey(cam='hst')

    for z in [4,5,6,7,8,10]:
        # Make sure we can recover filters over range in wavelength.
        hst_filt = hst.read_throughputs(filters='all')
        filt_p = get_filters_from_waves(z, hst_filt, picky=True)
        filt_np = get_filters_from_waves(z, hst_filt, picky=False)

        assert len(filt_p) <= len(filt_np), "{}, {}".format(filt_p, filt_np)

if __name__ == '__main__':
    test()
