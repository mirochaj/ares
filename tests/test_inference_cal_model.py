"""

test_inference_cal_model.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 09:28:21 EDT

Description: 

"""

import ares

def test():
    
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    
    # Test with variety of input arguments
    cal = ares.inference.CalibrateModel(fit_lf=[4], fit_beta=False,
        free_params_sfe=['norm', 'peak', 'slope-low', 'slope-high'],
        zevol_sfe=None,
        include_fduty=False, include_fdtmr=False, 
        save_sam=False, save_smf=False, save_lf=True, save_beta=False,
        save_sfrd=True, ztol=0.21)

    cal.base_kwargs = pars
    
    blobs = cal.blobs
    
    assert len(cal.parameters) == 4
    assert len(cal.guesses.keys()) == 4
    assert cal.free_params_dust == []
    
    
if __name__ == '__main__':
    test()


    
