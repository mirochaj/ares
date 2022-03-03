"""

Chemistry.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Sep 21 13:03:44 2012

Description:

Notes: If we want to parallelize over the grid, we'll need to use different
ODE integration routines, as scipy.integrate.ode is not re-entrant :(
Maybe not - MPI should be OK, multiprocessing should cause the problems.

"""

import copy
import numpy as np
from scipy.integrate import ode
from ..physics.Constants import k_B
from ..static.ChemicalNetwork import ChemicalNetwork

tiny_ion = 1e-12

class Chemistry(object):
    """ Class for evolving chemical reaction equations. """
    def __init__(self, grid, rt=False, atol=1e-8, rtol=1e-8, rate_src='fk94',
        recombination='B', interp_rc='linear'):
        """
        Create a chemistry object.

        Parameters
        ----------
        grid: ares.static.Grid.Grid instance
            Need this!
        rt: bool
            Use radiative transfer?

        """

        self.grid = grid
        self.rtON = rt

        self.chemnet = ChemicalNetwork(grid, rate_src=rate_src,
            recombination=recombination, interp_rc=interp_rc)

        # Only need to compute rate coefficients once for isothermal gas
        if self.grid.isothermal:
            self.rcs = \
                self.chemnet.SourceIndependentCoefficients(grid.data['Tk'])
        else:
            self.rcs = {}

        self.solver = ode(self.chemnet.RateEquations).set_integrator('lsoda',
            nsteps=1e4, atol=atol, rtol=rtol)

        self.solver._integrator.iwork[2] = -1

        # Empty arrays in the shapes we often need
        self.zeros_gridxq = np.zeros([self.grid.dims,
            len(self.grid.evolving_fields)])
        self.zeros_grid_x_abs = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.zeros_grid_x_abs2 = np.zeros_like(self.grid.zeros_grid_x_absorbers2)

    def Evolve(self, data, t, dt, **kwargs):
        """
        Evolve all cells by dt.

        Parameters
        ----------
        data : dictionary
            Dictionary containing elements for each field in the grid.
            Each element is itself a 1-D array of values.
        t : float
            Current time.
        dt : float
            Current time-step.
        """

        if self.grid.expansion:
            z = self.grid.cosm.TimeToRedshiftConverter(0, t, self.grid.zi)
            dz = dt / self.grid.cosm.dtdz(z)
        else:
            z = dz = 0

        if 'he_1' in self.grid.absorbers:
            i = self.grid.absorbers.index('he_1')
            self.chemnet.psi[...,i] *= data['he_2'] / data['he_1']

        # Make sure we've got number densities
        if 'n' not in data.keys():
            data['n'] = self.grid.particle_density(data, z)

        newdata = {}
        for field in data:
            newdata[field] = data[field].copy()

        if not kwargs:
            kwargs = self.rcs.copy()

        kwargs_by_cell = self._sort_kwargs_by_cell(kwargs)

        self.q_grid = np.zeros_like(self.zeros_gridxq)
        self.dqdt_grid = np.zeros_like(self.zeros_gridxq)

        # For debugging
        self.kwargs_by_cell = kwargs_by_cell

        # Loop over grid and solve chemistry
        for cell in range(self.grid.dims):

            # Construct q vector
            q = np.zeros(len(self.grid.evolving_fields))
            for i, species in enumerate(self.grid.evolving_fields):
                q[i] = data[species][cell]

            kwargs_cell = kwargs_by_cell[cell]

            if self.rtON:
                args = (cell, kwargs_cell['k_ion'], kwargs_cell['k_ion2'],
                    kwargs_cell['k_heat'], kwargs_cell['k_heat_lya'],
                    data['n'][cell], t)
            else:
                args = (cell, self.grid.zeros_absorbers,
                    self.grid.zeros_absorbers2, self.grid.zeros_absorbers,
                    0.0, data['n'][cell], t)

            self.solver.set_initial_value(q, 0.0).set_f_params(args).set_jac_params(args)

            self.solver.integrate(dt)

            self.q_grid[cell] = q.copy()
            self.dqdt_grid[cell] = self.chemnet.dqdt.copy()

            for i, value in enumerate(self.solver.y):
                newdata[self.grid.evolving_fields[i]][cell] = self.solver.y[i]

        # Compute particle density
        newdata['n'] = self.grid.particle_density(newdata, z - dz)

        # Fix helium fractions if approx_He==True.
        if self.grid.pf['include_He']:
            if self.grid.pf['approx_He']:
                newdata['he_1'] = newdata['h_1']
                newdata['he_2'] = newdata['h_2']
                newdata['he_3'] = np.zeros_like(newdata['h_1'])

        return newdata

    def _sort_kwargs_by_cell(self, kwargs):
        """
        Convert kwargs dictionary to list.

        Entries correspond to cells, a dictionary of values for each.
        """

        # Organize by cell
        kwargs_by_cell = []
        for cell in range(self.grid.dims):
            new_kwargs = {}
            for key in kwargs.keys():
                new_kwargs[key] = kwargs[key][cell]

            kwargs_by_cell.append(new_kwargs)

        return kwargs_by_cell
