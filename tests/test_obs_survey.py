"""

tests/test_obs_survey.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Tue 21 Dec 2021 14:38:11 EST

Description:

"""

import numpy as np
from ares.obs import Survey

def test():
    hst = Survey(cam='hst')

    zarr = [4, 5, 6, 7, 8, 10]
    drops = ['F435W', 'F606W', 'F775W', 'F850LP', 'F105W', 'F125W']
    for (z, drop) in zip(zarr, drops):

        fred, fblu = hst.get_dropout_filter(z,
            drop_wave=912. if z < 6 else 1216., skip=['F110W'])

        assert fred == drop, \
            "Dropout filter at z={} should be {}, got {}".format(z, drop, fred)


if __name__ == '__main__':
    test()
