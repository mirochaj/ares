"""

PointSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:02 MDT 2014

Description:

"""

import numpy as np
from ..util import ProgressBar
from .GasParcel import GasParcel
from ..solvers import RadialField
from ..util.PrintInfo import print_1d_sim
from ..util.ReadData import _sort_history
from ..analysis.RaySegment import RaySegment as AnalyzeRay

class RaySegment(AnalyzeRay):
    """
    Propagate radiation along a ray!
    """
    def __init__(self, **kwargs):
        """
        Initialize a RaySegment object.
        """

        self.parcel = GasParcel(**kwargs)

        self.pf = self.parcel.pf
        self.grid = self.parcel.grid

        # Initialize generator for gas parcel
        self.gen = self.parcel.step()

        # Rate coefficients for initial conditions
        self.parcel.update_rate_coefficients(self.grid.data)
        self._set_radiation_field()

        # Print info to screen
        if self.pf['verbose']:
            print_1d_sim(self)

    @property
    def info(self):
        print_1d_sim(self)

    def save(self):
        pass

    def save_tables(self, prefix=None):
        """
        Save tables of rate coefficients (functions of column density).

        Parameters
        ----------
        prefix : str
            Prefix of output files.

        """

        Ns = len(self.field.sources)
        if Ns > 1:
            p = ['{0!s}_src_{1}'.format(prefix, str(i).zfill(2)) for i in range(Ns)]
        else:
            p = [prefix]

        for i, src in enumerate(self.field.sources):
            src.tab.save(prefix=p[i])

    def reset(self, **kwargs):
        del sel.gen
        self.gen = self.parcel.step()

    def _set_radiation_field(self):

        if not self.pf['radiative_transfer']:
            return

        self.field = RadialField(self.grid, **self.pf)

    def update_rate_coefficients(self, data, **RCs):
        self.parcel.update_rate_coefficients(data, **RCs)

    def run(self):
        """
        Run simulation from start to finish.

        Returns
        -------
        Nothing: sets `history` attribute.

        """

        tf = self.pf['stop_time'] * self.pf['time_units']

        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        pb.start()

        all_t = []
        all_data = []
        for t, dt, data in self.step():

            # Compute ionization / heating rate coefficient
            RCs = self.field.update_rate_coefficients(data, t)

            # Re-compute rate coefficients
            self.update_rate_coefficients(data, **RCs)

            # Save data
            all_t.append(t)
            all_data.append(data.copy())

            if t >= tf:
                break

            pb.update(t)

        pb.finish()

        to_return = _sort_history(all_data)
        to_return['t'] = np.array(all_t)

        self.history = to_return

        return to_return

    def step(self):
        """
        Evolve properties of gas parcel in time.
        """

        #if data is None:
        #    data = self.grid.data.copy()
        #if t == 0:
        #    dt = self.pf['time_units'] * self.pf['initial_timestep']
        #if tf is None:

        t = 0
        tf = self.pf['stop_time'] * self.pf['time_units']

        while t < tf:
            yield next(self.gen)
