# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

##
# @package superposed
# @version V1
# @brief Function that generates a random relief with a number of scales defined on the "iterations" line
# in the conf_terrain.csv file
##



import numpy as np
import random
import math


def superposed(config):
    # first terrain from random
    z_terrain_naive = terrain_naive(config.N_x+1)  # N_x + 1 from 0 to N_x (included)

    # smoothing via interpolation and multiscale!
    z_terrain_interp = terrain_superposition(z_terrain_naive, iterations=config.iterations)
    return z_terrain_interp


# maps the value v from old range [ol, lh] to new range [nl, nh]
def mapv(v_old, ol, oh, nl, nh):
    v_new = nl + (v_old * ((nh-nl) / (oh - ol)))
    return v_new


# returns a list of integers representing heights at each point.
def terrain_naive(count):
    heights = [mapv(random.random(), 0, 1, 0, 100) for i in range(count)]
    return heights


# cosine interpolation returns the intermediate point between a and b. mu is distance from a
def cosp(a, b, mu):
    mu2 = (1 - math.cos(mu * math.pi)) / 2
    interp = a * (1 - mu2) + b * mu2
    return interp


# interpolate a naive terrain with cosine interpolation
def terrain_interp(naive_terrain, sample=4):
    terrain = []

    # get every sample point from the naive terrain
    sample_points = naive_terrain[::sample]

    # loop on every point in sample point denoting
    for ii_s in range(len(sample_points)):

        # add current peak (sample point) to terrain
        terrain.append(sample_points[ii_s])

        # fill in "sample-1" number of intermediary points using cosine interp
        for jj in range(sample-1):
            # compute relative distance from the left
            mu = (jj + 1)/sample

            # compute interpolated point at relative distance mu
            a = sample_points[ii_s]
            b = sample_points[(ii_s + 1) % len(sample_points)]
            v = cosp(a, b, mu)

            # add an interpolated points
            terrain.append(v)

    return terrain


# Superposition several multiscale "naive" terrains
def terrain_superposition(naive_terrain, iterations=8):
    terrains = []

    # holds the sum of weights for normalisation
    weight_sum = 0

    # for every iteration
    for ii in range(iterations, 0, -1):
        terrain = []

        # compute the scaling factor (weight)
        weight = 1 / 2 ** (ii-1)

        # compute sampling frequency suggesting every 'sample'th point to be picked from the naive terrain
        sample = 1 << (iterations - ii)

        # get the sample points
        sample_points = naive_terrain[::sample]

        weight_sum += weight

        # loop on every point in sample point
        for ii_s in range(len(sample_points)):

            # add scaled current peak (sample point) to terrain
            terrain.append(weight * sample_points[ii_s])

            # fill in "sample-1" number of intermediary points using cosine interp
            for jj in range(sample - 1):
                # compute relative distance from the left
                mu = (jj + 1) / sample

                # compute interpolated point at relative distance mu
                a = sample_points[ii_s]
                b = sample_points[(ii_s + 1) % len(sample_points)]
                v = cosp(a, b, mu)

                # add an interpolated points
                terrain.append(weight * v)

        # append this terrain to the list for preparing the superposition
        terrains.append(terrain)

    # perform superposition and normalisation of terrains to obtain the final terrain
    final_terrain = [sum(x)/weight_sum for x in zip(*terrains)]

    return final_terrain
