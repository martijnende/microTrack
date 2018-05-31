from __future__ import print_function

import numpy as np
from scipy.spatial import Delaunay, ConvexHull


class correlate:

    def __init__(self):
        pass

    def log_likelihood(self, a, b):
        """
        Calculate log likelihood of a correlating with b, given data,
        assuming Gaussian normal probability density function
        """
        logl = -0.5*(np.log(2*np.pi*self.sigma_sq) + ((a - b)**2) * self.inv_sigma_sq).sum()
        return logl

    def correlation_likelihoods(self, scanA, scanB, cols, sigma):
        """
        Correlate the particle data between two scans, for a given standard
        deviation (tolerance)
        """

        self.scanA = scanA
        self.scanB = scanB

        # Get variance for each data column
        self.sigma_sq = np.array([sigma[col] ** 2 for col in cols])
        self.inv_sigma_sq = 1.0/self.sigma_sq

        data_scanA = self.data[scanA][cols].values
        data_scanB = self.data[scanB][cols].values

        M = np.zeros((len(data_scanA), len(data_scanB)), dtype=np.float)

        if self.verbose:
            print("Start one-way correlation")

        for i, a in enumerate(data_scanA):
            for j, b in enumerate(data_scanB):
                M[i, j] = self.log_likelihood(a, b)

        self.M = np.exp(M)
        return True

    def joint_correlation(self):
        """
        Calculate the probabilities that particles are correlated from
        scan A to scan B and back
        """

        match_depth = self.match_depth
        M = self.M
        # Allocate space for two-way correlations
        E = np.zeros_like(M)

        # Indices to sort each row (descending order)
        top_ab = np.argsort(M, axis=1)[:, ::-1]
        # Indices to sort each column (descending order)
        top_ba = np.argsort(M.T, axis=1)[:, ::-1]

        if self.verbose:
            print("Start two-way correlation")

        for i, a in enumerate(top_ab):
            # Get top matches in scan B
            matches_b = a[:match_depth]
            # Calculate probability of each match
            prob_b = M[i, matches_b]
            prob_b /= prob_b.sum()

            # Get top matches in scan A for each match in scan B
            matches_a = top_ba[matches_b, :match_depth]

            # Loop over all top matches in scan A
            for j, b in enumerate(matches_a):
                # For each match in scan A, check if it contains the current grain
                # (in scan A)
                which_a = np.where(b == i)[0]
                # If contains current grain in scan A
                if len(which_a) > 0:
                    which_a = which_a[0]
                    # Get the index of match in scan B
                    which_b = matches_b[j]
                    prob_a = M[b, which_b]
                    prob_a /= prob_a.sum()
                    prob_a = prob_a[which_a]
                    E[i, which_b] = prob_a * prob_b[j]

        E_prob_max = np.sort(E, axis=1)[:, :-2:-1].reshape(-1)
        select_grains = (E_prob_max > self.min_correlation_prob)

        E_inds_max = np.argsort(E, axis=1)[:, :-2:-1].reshape(-1)

        self.correlatedA = np.arange(M.shape[0])[select_grains]
        self.correlatedB = E_inds_max[select_grains]
        self.uncorrelatedA = np.arange(M.shape[0])[~select_grains]
        # self.uncorrelatedB = E_inds_max[~select_grains]
        corrB_inds = self.data[self.scanB].index.isin(self.correlatedB)
        self.uncorrelatedB = self.data[self.scanB].index[~corrB_inds]

        n_corr = np.sum(select_grains)
        n_uncorr = np.sum(~select_grains)

        if self.verbose:
            print("Finished two-way correlation:")
            print("  * correlated grains = %i (%.1f %%)" % (n_corr, 100*n_corr/M.shape[0]))
            print("  * uncorrelated grains = %i (%.1f %%)" % (n_uncorr, 100*n_uncorr/M.shape[0]))

        return True

    def calc_strains(self):

        scanA = self.scanA
        scanB = self.scanB

        indsA = self.correlatedA
        indsB = self.correlatedB

        data_scanA = self.data[scanA]
        data_scanB = self.data[scanB]
        data_scanA["local_vol"] = 0.0
        data_scanB["local_vol"] = 0.0

        coords = self.strain_coords

        pointsA = data_scanA[coords]
        pointsB = data_scanB[coords]

        triA = Delaunay(data_scanA[coords].loc[indsA])

        local_vol = np.zeros((len(indsA), 2), dtype=float)

        if self.verbose:
            print("Calculating local strains")

        for i, region in enumerate(triA.simplices):
            reg_pointsA = pointsA.loc[indsA[region]]
            reg_pointsB = pointsB.loc[indsB[region]]
            volA = ConvexHull(reg_pointsA).volume
            volB = ConvexHull(reg_pointsB).volume
            local_vol[region, 0] += volA
            local_vol[region, 1] += volB

        volsA = local_vol[:, 0]
        volsB = local_vol[:, 1]
        self.strains = 2 * (volsA - volsB) / (volsA + volsB)

        return True
