"""

RadialField.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Feb 19 13:38:28 MST 2015

Description:

"""

import copy
import numpy as np
from ..sources import Composite
from ..util import ParameterFile
from ..static import LocalVolume
from ..physics.Constants import erg_per_ev, E_LyA, ev_per_hz

class RadialField:
    def __init__(self, grid, **kwargs):
        self.grid = grid
        self.pf = ParameterFile(**kwargs)

        # Just so we can access more easily from within RaySegment
        self.sources = Composite(self.grid, **self.pf).all_sources

        # Create instance to compute rate coefficients
        self.volume = LocalVolume(grid, self.sources, **kwargs)

    def update_rate_coefficients(self, data, t):
        """
        Compute rate coefficients for ionization / heating.
        """

        self.update_column_densities(data)
        return self.volume.update_rate_coefficients(data, t, self)

    def update_column_densities(self, data):
        """
        Compute column densities cumulatively and on a cell-by-cell basis.

        Parameters
        ----------
        data : dict
            Dataset for a single RaySegment snapshot.

        """

        # Column density to cells (N) and of cells (Nc)
        self.N, self.logN, self.Nc = self.grid.ColumnDensity(data)

        # Column densities (of all absorbers) sorted by cell
        # (i.e. an array with shape = grid cells x # of absorbers)
        self.N_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
        self.Nc_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
        for i, absorber in enumerate(self.grid.absorbers):
            self.N_by_cell[...,i] = self.N[absorber]
            self.Nc_by_cell[...,i] = self.Nc[absorber]

        self.logN_by_cell = np.log10(self.N_by_cell)
        self.logNc_by_cell = np.log10(self.N_by_cell)

        # Number densities
        self.n = {}
        for absorber in self.grid.absorbers:
            self.n[absorber] = data[absorber] * self.grid.x_to_n[absorber]

        # Compute column densities up to and of cells
        if self.pf['photon_conserving']:
            self.NdN = self.grid.N_absorbers \
                * [np.zeros_like(self.grid.zeros_grid_x_absorbers)]
            for i, absorber in enumerate(self.grid.absorbers):
                tmp = self.N_by_cell.copy()
                tmp[..., i] += self.Nc_by_cell[..., i]
                self.NdN[i] = tmp
                del tmp

            self.logNdN = np.log10(self.NdN)

    def update_photon_packets(self, newdata, t, dt, h): # pragma: no cover
        """
        Solver for finite speed-of-light radiation field

        PhotonPackage guide:
            pack = [EmissionTime, EmissionTimeInterval, ncolHI, ncolHeI, ncolHeII, Energy]
        """

        # Set up timestep array for use on next cycle
        if self.AdaptiveGlobalStep:
            dtphot = 1.e50 * np.ones_like(self.grid)
        else:
            dtphot = dt

        Lbol = self.rs.BolometricLuminosity(t)

        # Photon packages going from oldest to youngest - will have to create it on first timestep
        if t == 0:
            packs = []
        else:
            packs = list(self.data['PhotonPackages'])

        # Add one for this timestep
        packs.append(np.array([t, dt, neglible_column, neglible_column, neglible_column, Lbol * dt]))

        # Loop over photon packages, updating values in cells: data -> newdata
        for j, pack in enumerate(packs):
            t_birth = pack[0]
            r_pack = (t - t_birth) * c        # Position of package before evolving photons
            r_max = r_pack + dt * c           # Furthest this package will get this timestep

            # Cells we need to know about - not necessarily integer
            cell_pack = (r_pack - self.R0) * self.GridDimensions / self.pf['LengthUnits']
            cell_pack_max = (r_max - self.R0) * self.GridDimensions / self.pf['LengthUnits'] - 1
            cell_t = t

            Lbol = pack[-1] / pack[1]

            # Advance this photon package as far as it will go on this global timestep
            while cell_pack < cell_pack_max:

                # What cell are we in
                if cell_pack < 0:
                    cell = -1
                else:
                    cell = int(cell_pack)

                if cell >= self.GridDimensions:
                    break

                # Compute dc (like dx but in fractional cell units)
                # Really how far this photon can go in this step
                if cell_pack % 1 == 0:
                    dc = min(cell_pack_max - cell_pack, 1)
                else:
                    dc = min(math.ceil(cell_pack) - cell_pack, cell_pack_max - cell_pack)

                # We really need to evolve this cell until the next photon package arrives, which
                # is probably longer than a cell crossing time unless the global dt is vv small.
                if (len(packs) > 1) and ((j + 1) < len(packs)):
                    subdt = min(dt, packs[j + 1][0] - pack[0])
                else:
                    subdt = dt

                # If photons haven't hit first cell interface yet, evolve in time
                if cell < 0:
                    cell_pack += dc
                    cell_t += subdt
                    continue

                # Current radius in code units
                r = cell_pack * self.pf['LengthUnits'] / self.pf['GridDimensions']

                # These quantities will be different (in general) for each step
                # of the while loop
                n_e = newdata["ElectronDensity"][cell]
                n_HI = newdata["HIDensity"][cell]
                n_HII = newdata["HIIDensity"][cell]
                n_HeI = newdata["HeIDensity"][cell]
                n_HeII = newdata["HeIIDensity"][cell]
                n_HeIII = newdata["HeIIIDensity"][cell]
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII

                # Read in ionized fractions for this cell
                x_HI = n_HI / n_H
                x_HII = n_HII / n_H
                x_HeI = n_HeI / n_He
                x_HeII = n_HeII = n_He
                x_HeIII = n_HeIII = n_He

                # Compute mean molecular weight for this cell
                mu = 1. / (self.cosm.X * (1. + x_HII) + self.cosm.Y * (1. + x_HeII + x_HeIII) / 4.)

                # Retrieve path length through this cell
                dx = self.dx[cell]

                # Crossing time
                dct = self.CellCrossingTime[cell]

                # For convenience
                nabs = np.array([n_HI, n_HeI, n_HeII])
                nion = np.array([n_HII, n_HeII, n_HeIII])
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII
                n_B = n_H + n_He + n_e

                # Compute internal energy for this cell
                T = newdata["Temperature"][cell]
                E = 3. * k_B * T * n_B / mu / 2.

                q_cell = [n_HII, n_HeII, n_HeIII, E]

                # Add columns of this cell
                packs[j][2] += newdata['HIDensity'][cell] * dc * dx
                packs[j][3] += newdata['HeIDensity'][cell] * dc * dx
                packs[j][4] += newdata['HeIIDensity'][cell] * dc * dx
                ncol = np.log10(packs[j][2:5])

                ######################################
                ######## Solve Rate Equations ########
                ######################################

                # Retrieve indices used for interpolation
                indices = self.coeff.Interpolate.GetIndices(ncol)

                # Retrieve coefficients and what not.
                args = [nabs, nion, n_H, n_He, n_e]
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx * dc, t, self.z, cell))

                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args

                ######################################
                ######## Solve Rate Equations ########
                ######################################

                tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations,
                    q_cell, cell_t, cell_t + subdt, self.z, self.z - self.dz, None, h, *args)

                # Unpack results of coupled equations - Remember: these are lists and we only need the last entry
                newHII, newHeII, newHeIII, newE = qnew

                # Weight by volume if less than a cell traversed
                if dc < 1:
                    dnHII = newHII - newdata['HIIDensity'][cell]
                    dnHeII = newHeII - newdata['HeIIDensity'][cell]
                    dnHeIII = newHeIII - newdata['HeIIIDensity'][cell]
                    dV = self.coeff.ShellVolume(r, dx * dc) / self.coeff.ShellVolume(self.r[cell], dx)
                    newHII = newdata['HIIDensity'][cell] + dnHII * dV
                    newHeII = newdata['HeIIDensity'][cell] + dnHeII * dV
                    newHeIII = newdata['HeIIIDensity'][cell] + dnHeIII * dV

                # Calculate neutral fractions
                newHI = n_H - newHII
                newHeI = n_He - newHeII - newHeIII

                # Convert from internal energy back to temperature
                newT = newE * 2. * mu / 3. / k_B / n_B

                # Store data
                newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                    self.tau_all[cell], odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi,
                    omega, hubble, compton)

                cell_pack += dc
                cell_t += subdt

                ######################################
                ################ DONE ################
                ######################################

        # Adjust timestep for next cycle
        if self.AdaptiveGlobalStep:
            n_HI = newdata['HIDensity']
            n_HII = newdata['HIIDensity']
            n_H_all = n_HI + n_HII
            n_HeI = newdata['HeIDensity']
            n_HeII = newdata['HeIIDensity']
            n_HeIII = newdata['HeIIIDensity']
            n_He_all = n_HeI + n_HeII + n_HeIII
            n_e_all = n_HII + n_HeII + 2 * n_HeIII
            T = newdata['Temperature']
            n_B_all = n_H_all + n_He_all + n_e_all

            ncol_HI = np.roll(np.cumsum(n_HI * self.dx), 1)
            ncol_HeI = np.roll(np.cumsum(n_HeI * self.dx), 1)
            ncol_HeII = np.roll(np.cumsum(n_HeII * self.dx), 1)
            ncol_HI[0] = ncol_HeI[0] = ncol_HeII[0] = neglible_column
            ncol = np.transpose(np.log10([ncol_HI, ncol_HeI, ncol_HeII]))

            tau = self.ComputeOpticalDepths([ncol_HI, ncol_HeI, ncol_HeII])

            for cell in self.grid:
                r = self.r[cell]
                dx = self.dx[cell]
                nabs = np.array([n_HI[cell], n_HeI[cell], n_HeII[cell]])
                nion = np.array([n_HII[cell], n_HeII[cell], n_HeIII[cell]])

                indices = self.coeff.Interpolate.GetIndices(ncol[cell])

                args = [nabs, nion, n_H_all[cell], n_He_all[cell], n_e_all[cell]]
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol[cell], T[cell], dx, t, self.z))

                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args

                dtphot[cell] = self.control.ComputePhotonTimestep(tau[:,cell],
                    nabs, nion, ncol[cell], n_H, n_He,
                    n_e, n_B_all[cell], Gamma, gamma, Beta, alpha, k_H, zeta,
                    eta, psi, xi, omega, hubble, compton, T[cell], self.z, dt)

                if self.pf['LightCrossingTimeRestrictedTimestep']:
                    dtphot[cell] = min(dtphot[cell],
                        self.LightCrossingTimeRestrictedTimestep * self.CellCrossingTime[cell])

        # Update photon packages
        newdata['PhotonPackages'] = np.array(self.UpdatePhotonPackages(packs, t + dt))

        return newdata, dtphot
