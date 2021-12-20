"""

test_util_misc.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Sun 19 Dec 2021 18:52:05 EST

Description:

"""

import numpy as np
from ares.util.Misc import get_cmd_line_kwargs, get_attribute, split_by_sign

def test():

    sys_argv = ['scriptname', 'int_var=3', 'str_var=hello',
        'mix_var=ax323', 'bool_var=True', 'None_var=None',
        'float_var=12.3', 'special_var=23_45']

    kwargs = get_cmd_line_kwargs(sys_argv)

    assert kwargs['int_var'] == 3
    assert kwargs['str_var'] == 'hello'
    assert kwargs['mix_var'] == 'ax323'
    assert kwargs['bool_var'] is True
    assert kwargs['None_var'] is None
    assert kwargs['float_var'] == 12.3

    x = np.arange(0, 6 * np.pi, 500)
    y = np.sin(x)

    xch, ych = split_by_sign(x, y)
    for i, (xc, yc) in enumerate(zip(xch, ych)):
        assert np.all(np.sign(xc) == np.sign(xc[0]))
        assert np.all(np.sign(yc) == np.sign(yc[0]))



if __name__ == '__main__':
    test()
