from __future__ import print_function

import numpy as np
import pandas as pd

from microtrack.correlate import correlate
from microtrack.plot_functions import plot_functions


class read_data(correlate, plot_functions):

    # How many grains to compare recursively
    match_depth = 3
    # Minimum joint probability for making two-way correlation
    min_correlation_prob = 0.8
    # Dimensionality of strain calculations (2D/3D)
    strain_coords = ["x", "y", "z"]
    # Request output (print statements)
    verbose = True

    def __init__(self, filenames):
        correlate.__init__(self)
        plot_functions.__init__(self)
        col_names = ("theta", "phi", "twotheta", "twophi",
                     "x", "y", "z", "area", "vol", "d", "id")

        self.data = {}
        for name, file in filenames.items():
            data = pd.read_excel(file, names=col_names, header=1)
            midx = 0.5 * (data["x"].max() + data["x"].min())
            midy = 0.5 * (data["y"].max() + data["y"].min())
            data["r"] = np.sqrt((data["x"] - midx)**2 + (data["y"] - midy)**2)
            self.data[name] = data
        pass


if __name__ == "__main__":
    pass
