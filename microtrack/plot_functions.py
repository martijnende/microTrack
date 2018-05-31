from __future__ import print_function

# PyCharm optimise tends to remove the Axes3D, which are
# required by the 3D projection. Keep commented import
# for backup
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set(font_scale=1.2)


class plot_functions:

    def __init__(self):
        pass

    def plot_quiver2D(self):

        scanA = self.scanA
        scanB = self.scanB

        dataA = self.data[scanA][["x", "y", "z", "r"]]
        dataB = self.data[scanB][["x", "y", "z", "r"]]

        fig = plt.figure(figsize=(7, 8))
        ax = fig.gca(aspect="equal")

        for i in range(len(self.correlatedA)):
            pointA = dataA.loc[self.correlatedA[i]]
            pointB = dataB.loc[self.correlatedB[i]]
            dx, dy, dz, dr = (pointB.values - pointA.values)
            # dz /= pointA["z"]
            # dr /= pointA["r"]
            # scale = 500
            scale = 1
            ax.quiver(pointA["r"], pointA["z"], dr*scale, dz*scale,
                      alpha=0.5, color="C0",
                      scale_units="xy", angles="xy", scale=2)

        for i in range(len(self.uncorrelatedA)):
            pointA = dataA.loc[self.uncorrelatedA[i]]
            ax.scatter(pointA["r"], pointA["z"], c="r", s=5)

        for i in range(len(self.uncorrelatedB)):
            pointB = dataB.loc[self.uncorrelatedB[i]]
            ax.scatter(pointB["r"], pointB["z"], c="k", s=5)

        plt.xlabel(r"radius [$\mu$m]", labelpad=20)
        plt.ylabel(r"z-position [$\mu$m]", labelpad=20)
        plt.xlim((0.0, dataA["r"].max()))
        plt.ylim((dataA["z"].min(), dataA["z"].max()))
        plt.tight_layout()
        plt.show()

    def plot_quiver3D(self):

        scanA = self.scanA
        scanB = self.scanB

        dataA = self.data[scanA][["x", "y", "z"]]
        dataB = self.data[scanB][["x", "y", "z"]]

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        for i in range(len(self.correlatedA)):
            pointA = dataA.loc[self.correlatedA[i]]
            pointB = dataB.loc[self.correlatedB[i]]
            dx, dy, dz = (pointB.values - pointA.values)
            ax.quiver(pointA["x"], pointA["y"], pointA["z"], dx, dy, dz)

        for i in range(len(self.uncorrelatedA)):
            pointA = dataA.loc[self.uncorrelatedA[i]]
            ax.scatter(pointA["x"], pointA["y"], pointA["z"], c="r", s=5)

        for i in range(len(self.uncorrelatedB)):
            pointB = dataB.loc[self.uncorrelatedB[i]]
            ax.scatter(pointB["x"], pointB["y"], pointB["z"], c="k", s=5)

        plt.tight_layout()
        plt.show()

    def plot_strains3D(self):

        scan = self.scanA
        data = self.data[scan][["x", "y", "z"]].loc[self.correlatedA]
        strains = self.strains

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        p = ax.scatter(data["x"], data["y"], data["z"],
                       c=strains, vmin=-0.1, vmax=0.1,
                       cmap="coolwarm", s=10)
        fig.colorbar(p)
        plt.tight_layout()
        plt.show()


    def plot_strains2D(self, projection="cylindrical"):

        scan = self.scanA
        data = self.data[scan][["x", "y", "z", "r"]].loc[self.correlatedA]
        strains = self.strains*100
        smax = 10.0
        smin = -10.0

        Ngrid = 10
        mean_strain = np.zeros((Ngrid, Ngrid), dtype=float)
        gridz = np.linspace(data["z"].min(), data["z"].max(), Ngrid + 1)

        cmap = "coolwarm"

        if projection is "cartesian":
            mean_strain2 = np.zeros((Ngrid, Ngrid), dtype=float)
            gridx = np.linspace(data["x"].min(), data["x"].max(), Ngrid+1)
            gridy = np.linspace(data["y"].min(), data["y"].max(), Ngrid+1)

            for i, x in enumerate(gridx[:-1]):
                inds_x = (data["x"] > x) & (data["x"] <= gridx[i + 1])
                for j, z in enumerate(gridz[:-1]):
                    inds_z = (data["z"] > z) & (data["z"] <= gridz[j+1])
                    inds = inds_x & inds_z
                    mean_strain[i, j] = np.median(strains[inds])

            for i, y in enumerate(gridy[:-1]):
                inds_y = (data["y"] > y) & (data["y"] <= gridy[i + 1])
                for j, z in enumerate(gridz[:-1]):
                    inds_z = (data["z"] > z) & (data["z"] <= gridz[j + 1])
                    inds = inds_y & inds_z
                    mean_strain2[i, j] = np.median(strains[inds])

            ax = plt.subplot(121)
            ax.matshow(mean_strain, vmin=smin, vmax=smax, cmap=cmap, origin="lower")
            plt.xlabel("x")
            plt.ylabel("z")
            ax = plt.subplot(122)
            p = ax.matshow(mean_strain2, vmin=smin, vmax=smax, cmap=cmap, origin="lower")
            plt.xlabel("y")
            plt.colorbar(p)
        elif projection is "cylindrical":
            gridr = np.linspace(0.0, data["r"].max(), Ngrid+1)
            for i, r in enumerate(gridr[:-1]):
                inds_r = (data["r"] > r) & (data["r"] <= gridr[i + 1])
                for j, z in enumerate(gridz[:-1]):
                    inds_z = (data["z"] > z) & (data["z"] <= gridz[j+1])
                    inds = inds_r & inds_z
                    if inds.sum() > 0:
                        mean_strain[j, i] = np.median(strains[inds])

            extent = (0.0, data["r"].max(), gridz.min(), gridz.max())
            plt.figure(figsize=(7, 8))
            ax = plt.gca()
            p = ax.matshow(mean_strain, vmin=smin, vmax=smax, cmap=cmap,
                            origin="lower", extent=extent, interpolation="gaussian")
            ax.scatter(data["r"], data["z"], c=strains, s=20, cmap=cmap,
                       edgecolor="k", linewidth=1.0, vmin=smin, vmax=smax)
            cb = plt.colorbar(p)
            cb.set_label("Volumetric strain [%] \n (compaction positive)",
                         rotation=270, labelpad=35)
            plt.xlabel(r"radius [$\mu$m]", labelpad=20)
            plt.ylabel(r"z-position [$\mu$m]", labelpad=20)
            plt.xlim((0.0, data["r"].max()))
            plt.ylim((gridz.min(), gridz.max()))
            ax.xaxis.tick_bottom()

        plt.tight_layout()
        plt.show()
