"""

test_util_misc.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue 24 Mar 2020 21:31:12 EDT

Description: 

"""

import ares
import numpy as np

def test():
    """
    Run through miscellaneous functions and make sure they run to completion
    for a variety of cases.
    """

    fake_sys_argv = ['script_name', 'param1=a_string', 'param2=2', 'param3=4.0', 
        'param4=1e10', 'param5=True', 'param6=None', 'param7=[1,2,3]']
    kw = ares.util.Misc.get_cmd_line_kwargs(fake_sys_argv)
    kw_types = [type(kw[key]) for key in kw.keys()]
    # Check these somehow?
    
    ares_rev = ares.util.Misc.get_rev()
    
    
    n_nu_1 = ares.util.Misc.num_freq_bins(100)
    n_nu_2 = ares.util.Misc.num_freq_bins(200)
    assert n_nu_2 > n_nu_1

    #class FakeClass(object):
    #    self.a = 1
    #    self.b = 2
    #
    #result = ares.util.Misc.get_attribute('t)
    #assert s1    
    
    x = np.arange(100)
    y = np.sin(x)
    xch, ych = ares.util.Misc.split_by_sign(x, y)
    ct = 0
    for element in xch:
        ct += len(element)
    assert ct == len(x)
            

    
if __name__ == '__main__':
    test()
