import ares


def test():
    pars = ares.util.ParameterBundle('mirocha2017:base')

    orph = pars.orphans
    bkw  = pars.get_base_kwargs()
    
    assert pars.pqids == [0]
    assert pars.pqs == ['pop_fstar{0}']
    
    sfe_pars = pars.pars_by_pq(0)
        
if __name__ == '__main__':
    test()
    
    