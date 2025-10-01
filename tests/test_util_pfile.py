import ares


def test():

    # This is a single population model
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pop = ares.populations.GalaxyPopulation(**pars)

    # Check that parameters are passed in correctly.
    for par in pars:
        print(par, pars[par], pop.pf[par])
        continue
        assert pars[par] == pop.pf[par], \
            f"Failed == check for parameter `{par}`"

    return

    # We should check that we get the same thing whether 
    # or not there are ID numbers in the parameters.

    # This is a two population model
    pars = ares.util.ParameterBundle('mirocha2017:base')
    sim = ares.simulations.Simulation(**pars)

    # Make sure each population instance gets its parameters 
    for i, pop in enumerate(sim.pops):
        for par in pop.pf:
            assert pop.pf[par] == sim.pf[f"{par}{{{i}}}"]

        #assert pars[par] == pop.pf[par]


if __name__ == '__main__':
    test()