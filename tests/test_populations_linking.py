import ares

def test():
    pars = ares.util.ParameterBundle('mirocha2025:base')
    #pars.update(ares.util.ParameterBundle('testing:galaxies'))
    
    sim = ares.simulations.Simulation(**pars)
        
    assert sim.pops[0].get_smhm(z=0.1, Mh=1e10) == \
           sim.pops[2].get_smhm(z=0.1, Mh=1e10)
    
    assert sim.pops[1].get_smhm(z=0.1, Mh=1e10) == \
           sim.pops[3].get_smhm(z=0.1, Mh=1e10)
    
    assert sim.pops[0].get_sfr(z=0.1, Mh=1e10) == \
           sim.pops[2].get_sfr(z=0.1, Mh=1e10)
    
    assert sim.pops[1].get_sfr(z=0.1, Mh=1e10) == \
           sim.pops[3].get_sfr(z=0.1, Mh=1e10)
    
    # Slightly trickier
    assert sim.pops[0].get_focc(z=0.1, Mh=1e10) == \
      1. - sim.pops[1].get_focc(z=0.1, Mh=1e10)
    
if __name__ == '__main__':
    test()
