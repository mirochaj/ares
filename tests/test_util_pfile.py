import ares
from ares.util.ParameterFile import par_info

def test():

    # This is a single population model
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars.update(ares.util.ParameterBundle('testing:galaxies'))
    pop = ares.populations.GalaxyPopulation(**pars)
    # Check that parameters are passed in correctly.
    for par in pars:
        s = f"pars[{par}]={pars[par]} | pop.pf[{par}]={pop.pf[par]}"
        assert pars[par] == pop.pf[par], \
            f"Failed == check for parameter `{par}`: {s}"
        
    ##
    # Do again but through Simulation object
    sim = ares.simulations.Simulation(**pars)
    for par in pars:
        s = f"pars[{par}]={pars[par]} | sim.pops[0].pf[{par}]={pop.pf[par]}"
        assert pars[par] == sim.pops[0].pf[par], \
            f"Failed == check for parameter `{par}`: {s}"
    
    ##
    # This model has a few PQs. Check that they survived.
    assert pop.pf.Npqs == sim.pops[0].pf.Npqs
    assert sim.pf.Npqs == sim.pops[0].pf.Npqs
    
    # Check that each PQ has a full set of parameters?
    # Or, must the user make sure to get the full set?
    for pq in sim.pf.pqs:
        pqp = sim.pf.get_pq_pars(sim.pf[pq])
        # Should we check that each one has the full 35 default parameters?
    
    
    ##
    # Do this again if the user has added pop ID number?
    # For single population models, IDs are encouraged but not required.
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars.update(ares.util.ParameterBundle('testing:galaxies'))
    pars.num = 0
    pop = ares.populations.GalaxyPopulation(**pars)
    
    # In this case, just be careful in that the ID number 
    # will be present in `pars` parameters but not `pop.pf`.
    for par in pars:
    
        is_pop_or_pq = par.startswith('pop_') or par.startswith('pq_')
        # Check for pop parameter with ID number
        if is_pop_or_pq and '{' in par:
            par_noID = par[0:par.rfind('{')]
            s = f"pars[{par}]={pars[par]} |  pop.pf[{par}]={pop.pf[par_noID]}"
            assert pars[par] == pop.pf[par_noID], \
                f"Failed == check for parameter `{par}`: {s}"
        # Population parameter without ID number
        elif is_pop_or_pq:
            s = f"pars[{par}]={pars[par]} | pop.pf[{par}]={pop.pf[par]}"
            assert pars[par] == pop.pf[par], \
                f"Failed == check for parameter `{par}`: {s}"
        # General parameter. If this fails we've done something very wrong
        else:
            s = f"pars[{par}]={pars[par]} | pop.pf[{par}]={pop.pf[par]}"
            
            assert pars[par] == pop.pf[par], \
                f"Failed == check for parameter `{par}`: {s}"
    
    
    
    # This is a two population model
    pars = ares.util.ParameterBundle('mirocha2017:base')
    sim = ares.simulations.Simulation(**pars)
    
    assert sim.pf.Npops == 2
    assert sim.pf.Npqs == 1
    
    # Make sure each population instance gets its parameters 
    # and they match those in sim.pf.pfs.
    # Should population pars remain in sim.pf? I don't think they'll get used 
    # for anything...
    # One complication: linked parameters get taken care of in ParameterFile so 
    # will not match between pars and sim.pf (or sim.pops[x].pf)
    
    for par in pars:
        prefix, popid, pqpid = par_info(par)
        is_pop_or_pq = (popid is not None) or (pqpid is not None)
    
        for i, pop in enumerate(sim.pops):
    
            if not is_pop_or_pq:
                s = f"pars[{par}]={pars[par]} | sim.pops[{i}].pf[{par}]={pop.pf[par]}"
                assert pars[par] == pop.pf[par], \
                    f"Failed == check for parameter `{par}`: {s}"
                continue
            
            if popid != i:
                continue
    
            par_noID = par[0:par.rfind('{')]
            
            ##
            # Check for linked parameter 
            if type(pars[par]) == str:
                
                if pars[par].startswith(par_noID):
                    # This means population `popid` is linked to `popid2`
                    
                    prefix2, popid2, pqpid2 = par_info(pars[par])
                    assert sim.pops[popid2].pf[par_noID] == sim.pops[popid].pf[par_noID]
    
                    continue
    
            ##
            # Unlinked parameters (just numbers or w/e, much more common)
            s = f"pars[{par}]={pars[par]} | sim.pops[{i}].pf[{par_noID}]={pop.pf[par_noID]}"
            assert pars[par] == pop.pf[f"{par_noID}"], \
                f"Failed == check for parameter `{par}`: {s}"
            
            # Simulation instance will still have ID numbers
            s = f"sim.pf.pfs[{i}][{par_noID}]={pars[par]} | sim.pops[{i}].pf[{par_noID}]={pop.pf[par_noID]}"
            assert sim.pf.pfs[i][par_noID] == pop.pf[f"{par_noID}"], \
                f"Failed == check for parameter `{par}`: {s}"
                        


if __name__ == '__main__':
    test()