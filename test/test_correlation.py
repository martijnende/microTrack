from __future__ import print_function

import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
from numpy.testing import assert_allclose, assert_array_equal

from microtrack.correlate import correlate

inst = correlate()
inst.match_depth = 3
inst.min_correlation_prob = 0.8
inst.data = {}
inst.verbose = False
inst.strain_coords = ["x", "y"]


class test_correlation(unittest.TestCase):

    rtol = 1e-6

    @staticmethod
    def generate_synthetic():
        """Generate synthetic scan data"""

        N = 20

        np.random.seed(0)
        rand = np.random.rand(N, 4)

        p = pd.DataFrame()
        p["vol"] = rand[:, 0]*1e6
        p["curt"] = 1.0 + 0.2*rand[:, 1]
        p["x"] = rand[:, 2]
        p["y"] = rand[:, 3]
        p["z"] = 0.0

        p2 = p.copy()
        p2["vol"] += 1e4 * (np.random.rand(N) - 0.5)
        p2["x"] += 0.2 * (np.random.rand(N) - 0.5)
        p2["y"] += 0.5 * (np.random.rand(N) - 0.5)

        return p, p2

    def print_name(self, name):
        """Print out the class and method names during testing"""
        print("__%s__.%s... " % (self.__class__.__name__, name), end="")

    def test_likelihood(self):
        """Check that the log-likelihood function is Gaussian"""

        self.print_name(sys._getframe().f_code.co_name)

        a = 0.0
        b = np.linspace(-1.0, 1.0, int(1e2))
        inst.sigma_sq = 0.1
        inst.inv_sigma_sq = 1.0/inst.sigma_sq

        pdf = np.exp(np.array([inst.log_likelihood(a, bi) for bi in b]))
        pdf_true = stat.norm.pdf(b, loc=a, scale=np.sqrt(inst.sigma_sq))

        assert_allclose(pdf, pdf_true, 1e-12)

        print("OK")

    def test_twoway(self):
        """Visually test two-way correlations"""

        self.print_name(sys._getframe().f_code.co_name)

        p1, p2 = self.generate_synthetic()

        inst.data["set1"] = p1
        inst.data["set2"] = p2

        cols = ["vol", "x", "y"]
        sigma = {
            "vol": 1e3,
            "curt": 0.001,
            "x": 0.005,
            "y": 0.005,
        }

        inst.correlation_likelihoods("set1", "set2", cols, sigma)
        inst.joint_correlation()

        assert_array_equal(inst.correlatedA, inst.correlatedB)

        correlatedB = p2.index.isin(inst.correlatedB)
        uncorrelatedB = p2.index[~correlatedB]

        for i in range(len(inst.correlatedA)):
            pA = p1[["x", "y"]].loc[inst.correlatedA[i]]
            pB = p2[["x", "y"]].loc[inst.correlatedB[i]]
            dx, dy = (pB.values - pA.values)
            plt.quiver(pA["x"], pA["y"], dx, dy, scale_units="xy", angles="xy", scale=1)
            plt.plot(pA["x"], pA["y"], "ko")
            plt.plot(pB["x"], pB["y"], "ko")
            plt.text(pA["x"]+0.02, pA["y"], "%i" % inst.correlatedA[i])

        for a in inst.uncorrelatedA:
            pA = p1[["x", "y"]].loc[a]
            plt.plot(pA["x"], pA["y"], "ro")
            plt.text(pA["x"] + 0.02, pA["y"], "%i" % a)

        for b in uncorrelatedB:
            pB = p2[["x", "y"]].loc[b]
            plt.plot(pB["x"], pB["y"], "co")
            plt.text(pB["x"] + 0.02, pB["y"], "%i" % b)

        plt.show()

        print("OK")

    def test_strains(self):
        """Check if strain calculations are correct"""

        self.print_name(sys._getframe().f_code.co_name)

        p1, p2 = self.generate_synthetic()
        p2 = p1.copy()
        p2["x"] += 0.1
        p2["y"] += 0.1
        p2 = p2.sample(frac=1).reset_index(drop=True)

        inst.data["set1"] = p1
        inst.data["set2"] = p2

        cols = ["vol", "x", "y"]
        sigma = {
            "vol": 1e3,
            "curt": 0.001,
            "x": 0.01,
            "y": 0.01,
        }

        # Test strain = zero for rigid body displacement
        # when correlation is perfect
        inst.correlation_likelihoods("set1", "set2", cols, sigma)
        inst.joint_correlation()
        inst.calc_strains()

        assert_allclose(inst.strains, np.zeros_like(inst.strains), atol=1e-12)

        # Test strain = zero for rigid body displacement
        # when correlation is imperfect => skip this, does not give zero strain
        # even when correlation works as intended

        # Calc strain for simple geometric displacement (shear of box)
        cols = ["vol"]

        p1 = pd.DataFrame()
        p1["vol"] = np.arange(4)*1e6
        p1["x"] = np.array([0, 1, 0, 1])
        p1["y"] = np.array([0, 0, 1, 1])
        p1["z"] = 0.0

        p2 = p1.copy()
        p2["x"] = np.array([0, 1, 0.5, 1.5])
        p2 = p2.sample(frac=1).reset_index(drop=True)

        inst.data["set1"] = p1
        inst.data["set2"] = p2
        inst.correlation_likelihoods("set1", "set2", cols, sigma)
        inst.joint_correlation()
        inst.calc_strains()

        assert_allclose(inst.strains, np.zeros_like(inst.strains), atol=1e-12)

        # Calc strain for simple geometric displacement (moving one corner)
        p2 = p1.copy()
        p2["x"] = np.array([-1, 1, 0, 1])
        p2 = p2.sample(frac=1).reset_index(drop=True)
        inst.data["set2"] = p2
        inst.correlation_likelihoods("set1", "set2", cols, sigma)
        inst.joint_correlation()
        inst.calc_strains()

        assert_allclose(inst.strains.min(), -2.0/3.0, atol=1e-12)
        assert_allclose(inst.strains.max(), 0.0, atol=1e-12)
        assert_allclose(inst.strains.sum(), -22.0/15.0, atol=1e-12)

        print("OK")
