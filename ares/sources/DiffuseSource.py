"""

DiffuseSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jul  8 10:34:59 MDT 2013

Description: 

"""

class DiffuseSource:
    def __init__(self, rb):
        """
        Needs to know if IGM or HII region grid patch.
        """
        self.rb = rb # RadiationBackground instance
        self.SourcePars = {}
        self.SourcePars['type'] = 'diffuse'
        self.pf = self.rb.pf

    def SourceOn(self, t):
        return True

    def ionization_rate(self, z, species=0, **kwargs):
        if kwargs['igm']:
            return self.rb.igm.IonizationRateIGM(z, species=species, **kwargs)
        else:
            return self.rb.igm.IonizationRateHIIRegions(z, species=species, 
                **kwargs)

    def secondary_ionization_rate(self, z, species=0, **kwargs):
        if kwargs['igm']:
            return self.rb.igm.SecondaryIonizationRateIGM(z, species=species, 
                **kwargs)
        else:
            return 0.0

    def heating_rate(self, z, species=0, **kwargs):
         # This gets called twice per source per timestep
        if kwargs['igm']:
            heat = self.rb.igm.HeatingRate(z, species=species, **kwargs)
        else:
            heat = 0.0

        return heat
