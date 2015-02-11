"""

InlineAnalysis.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jan 12 11:04:33 MST 2015

Description: 

"""

import re
import numpy as np
from ..util.Misc import tau_CMB
from scipy.interpolate import interp1d
from .TurningPoints import TurningPoints
from ..physics.Constants import ev_per_hz, rhodot_cgs
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts

class InlineAnalysis:
    def __init__(self, sim):
        self.sim = sim
        self.pf = self.sim.pf
        self.history = self.sim.history
        
        z = np.array(self.history['z'])
        self.zmin = z.min()
        self.zmax = z.max()
        
        if self.pf['inline_analysis'] is not None and \
           (not self.pf['auto_generate_blobs']):
            self.blob_names, self.blob_redshifts = self.pf['inline_analysis']
        elif self.pf['auto_generate_blobs']:
            self.blob_names, self.blob_redshifts = self.generate_blobs()
    
        self.need_extrema = 0
        for tp in list('BCD'):
            if tp in self.blob_redshifts:
                self.need_extrema += 1
                
    def generate_blobs(self):
        """
        Auto-generate blobs for inline analysis.

        Returns
        -------
        Names and redshifts of blobs.

        """
        
        # First, figure out which rate coefficients to save
        # Automatically figure out which populations need what
        
        blob_names = []
        for i, pop in enumerate(range(self.pf.Npops)):
            
            if hasattr(self.sim, 'pops'):
                pop = self.sim.pops.pops[i]
            else:
                pop = self.sim
            
            if self.pf.Npops > 1:
                suffix = '{%i}' % i
            else:
                suffix = ''
                
            # Lyman-alpha emission
            if pop.pf['is_lya_src']:
                blob_names.append('Ja%s' % suffix)
                
            # SFRD
            if not pop.pf['tanh_model']:
                blob_names.append('sfrd%s' % suffix)
                            
            species = ['h_1', 'he_1', 'he_2']
            for j, sp1 in enumerate(species):
                
                if j > 0 and (not self.pf['include_He']):
                    break
            
                blob_names.append('igm_%s' % sp1)
                
                if pop.pf['is_ion_src_cgm'] and j == 0:
                    blob_names.append('cgm_Gamma_%s%s' % (sp1, suffix))
                    
                if pop.pf['is_heat_src_igm']:
                    blob_names.append('igm_heat_%s%s' % (sp1, suffix))
                
                if pop.pf['is_ion_src_igm'] and (not pop.pf['tanh_model']):
                    blob_names.append('igm_Gamma_%s%s' % (sp1, suffix))
                else:
                    continue
            
                if not pop.pf['secondary_ionization']:
                    continue
                    
                if pop.pf['tanh_model']:
                    continue
                                        
                for k, sp2 in enumerate(species):
                    if k > 0 and (not self.pf['include_He']):
                        break
                        
                    blob_names.append('igm_gamma_%s_%s%s' % (sp1, sp2, suffix))
            
        tmp1 = _blob_names
        tmp1.extend(blob_names)
        
        tmp11 = list(np.unique(tmp1))
        names = map(str, tmp11)
        
        tmp2 = []
        for i, z in enumerate(_blob_redshifts):
            if type(z) is str:
                tmp2.append(_blob_redshifts[i])
                continue
                
            if z >= self.sim.pf['final_redshift']:
                tmp2.append(_blob_redshifts[i])
                continue
        
        blob_names = names
        blob_redshifts = tmp2
         
        if self.pf['override_blob_names'] is not None:
            blob_names = self.pf['override_blob_names']
        if self.pf['override_blob_redshifts'] is not None:
            blob_redshifts = self.pf['override_blob_redshifts'] 
         
        return blob_names, blob_redshifts

    @property
    def turning_points(self):            
        if hasattr(self.sim, "turning_points"):
            self._turning_points = self.sim.turning_points
        elif not hasattr(self, '_turning_points') and self.need_extrema > 0:
            self._track = TurningPoints(inline=True, **self.pf)

            # Otherwise, find them. Not the most efficient, but it gets the job done
            if self.zmax < 70 and 'A' not in self._track.TPs:
                self._track.TPs.append('A')

            delay = self.pf['stop_delay']
            
            for i in range(len(self.history['z'])):
                if i < 10:
                    continue

                stop = self._track.is_stopping_point(self.history['z'][0:i], 
                    self.history['dTb'][0:i])

            self._turning_points = self._track.turning_points

        return self._turning_points
        
    def parse_redshifts(self):
        """
        Convert all redshifts to floats (e.g., turning points B, C, & D).
        """
        
        redshift = []
        for element in self.blob_redshifts:
            
            # Some "special" redshifts -- handle separately
            if type(element) is str:
                if element in ['eor_midpt', 'eor_overlap']:
        
                    raise ValueError('For some reason, eor_midpt etc. are causing problems for emcee / pickling.')
        
                    ihigh = np.argmin(np.abs(self.history['z'] \
                          - self.pf['first_light_redshift']))
                    interp = interp1d(self.history['cgm_h_2'][ihigh:],
                        self.history['z'][ihigh:])
        
                    try:
                        if element == 'eor_midpt':
                            zrei = interp(0.5)
                        else:
                            zrei = interp(0.99)

                        if type(zrei) != float:
                            zrei = float(zrei)

                    except ValueError:
                        zrei = np.inf

                    redshift.append(zrei)
        
                elif element not in self.turning_points:
                    redshift.append(np.inf)
                else:
                    redshift.append(self.turning_points[element][0])
            
            # Just a number, append and move on
            else:
                redshift.append(float(element))
                
        return redshift

    def run_inline_analysis(self):
        """
        Compute some quantities of interest.

        Example
        -------
        sim = ares.simulations.Global21cm(track_extrema=True, 
            inline_analysis=(['dTb'], list('BCD'))

        sim.run()

        zip(*sim.blobs)[0]  # are the brightness temperatures of B, C, and D
        sim.ztps            # redshifts

        """
    
        self.redshifts_fl = self.parse_redshifts()

        # Recover quantities of interest at specified redshifts
        output = []
        for j, field in enumerate(self.blob_names):
            
            m = re.search(r"\{([0-9])\}", field)

            if m is None:
                pop_specific = False
                pop_prefix = None

            else:
                pop_specific = True
                
                # Population ID number
                pop_num = int(m.group(1))
                
                # Pop ID including curly braces
                pop_prefix = field.strip(m.group(0))

            # Setup a spline interpolant
            if field in self.history:
                interp = interp1d(self.history['z'][-1::-1],
                    self.history[field][-1::-1])
            elif field == 'tau_e':
                tmp, tau_tmp = tau_CMB(self.sim)
                interp = interp1d(tmp, tau_tmp)

            elif field == 'curvature':
                tmp = []
                for element in self.blob_redshifts:

                    if element not in self.turning_points:
                        tmp.append(np.inf)
                        continue

                    if (type(element)) == str and (element != 'trans'):
                        tmp.append(self.turning_points[element][-1])
                    else:
                        tmp.append(np.inf)

                output.append(tmp)
                continue
            
            #elif field == 'Jlw':
            #    Jlw = self.integrated_fluxes()
            #    output.append(Jlw)
            #    continue
            
            elif (field == 'sfrd'):
                tmp = []
                for redshift in self.redshifts_fl:
                    if self.zmin <= z <= self.zmax:
                        sfrd = self.get_sfrd(redshift)
                    else:
                        sfrd = np.inf
                    tmp.append(sfrd)
                output.append(tmp)
                continue
            elif (pop_prefix == 'sfrd'):
                tmp = []
                for redshift in self.redshifts_fl:
                    if self.zmin <= z <= self.zmax:
                        sfrd = self.get_sfrd(redshift, num=pop_num)
                    else:
                        sfrd = np.inf
                    tmp.append(sfrd)
                output.append(tmp)
                continue

            # Go back and actually interpolate, save the result (for each z)
            tmp = []
            for i, z in enumerate(self.redshifts_fl):

                if z is None:
                    tmp.append(np.inf)
                    continue

                if self.zmin <= z <= self.zmax:
                    tmp.append(float(interp(z)))
                else:
                    tmp.append(np.inf)
                    
            output.append(tmp)
            
        # Reshape output so it's (redshift x blobs)
        self.blobs = np.array(zip(*output))
        
    def get_igm_quantity(self):
        pass    

    def integrated_fluxes(self, band='lw'):
        """
        Integrate flux in LW band (in future, maybe more general).
        """
        
        tmp = []
        for z in self.redshifts_fl:

            if (z is None) or (not (self.zmin <= z <= self.zmax)):
                tmp.append(np.inf)
                continue                

            # Bracket redshift of interest
            ilo = np.argmin(np.abs(self.sim.lwb_z - z))
            if self.sim.lwb_z[ilo] > z:
                ilo -= 1
            ihi = ilo + 1

            # Fudge
            if ihi > (len(self.sim.lwb_z) - 1):
                ilo -= 1
                ihi -= 1

            zlo, zhi = self.sim.lwb_z[ilo], self.sim.lwb_z[ihi]

            # Might have to worry about multiple POPS    

            # Compute integrated flux @ each pt.

            # Loop over radiation backgrounds
            for k, element in enumerate(self.sim._Jrb):
                
                if element is None:
                    continue
                
                junk_z, En, flux = self.sim._Jrb[k]

                # flux is (Nbands x Nz x NE)

                # Loop over bands

                Jlo = 0.0
                Jhi = 0.0
                for j, band in enumerate(En):
                    Jlo += np.trapz(flux[j][ilo], x=band) / ev_per_hz
                    Jhi += np.trapz(flux[j][ihi], x=band) / ev_per_hz

                # Subtract of 10.2-11.2 eV flux
                Earr = En[0]
                icut = np.argmin(np.abs(Earr - 11.18))
                Jlo -= np.trapz(flux[0][ilo][0:icut], x=En[0][0:icut]) / ev_per_hz
                Jhi -= np.trapz(flux[0][ihi][0:icut], x=En[0][0:icut]) / ev_per_hz
                
            Jz = np.interp(z, [zlo, zhi], [Jlo, Jhi])
            tmp.append(Jz)

        return tmp

    def get_sfrd(self, z, num=None):
        
        # Single-pop model
        if num is None:
            try:
                return self.sim.pops.pops[0].SFRD(z) * rhodot_cgs
            except SystemExit:
                return np.inf
            
        # Multi-pop model
        for i, pop in enumerate(self.sim.pops.pops):
            if i != num:
                continue
                
            try:
                sfrd = pop.SFRD(z) * rhodot_cgs
            except SystemExit:
                sfrd = np.inf    
            
            return sfrd
                
        
        
        
        
    