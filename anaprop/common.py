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
# @package anaprop/common.py
# @author Remi Douvenot - storca -
# @brief Geodesic computations and cache system for files processed by the anaprop module
# @warning
##

import numpy as np
from geographiclib.geodesic import Geodesic
from typing import Tuple
from os import getenv, makedirs, rmdir
from os.path import isfile, join
from platformdirs import user_cache_dir
import hashlib
import xarray as xr


def geodesic_line_coords(P: Tuple[float], Q: Tuple[float], N: int):
    """
    Returns the coordinates of the geodesic line formed of N points between points P and Q.
    P : (lat, long)
    Q : (lat, long)
    N : number of points

    returns : np.array formed of the N (lat, long) points
    """
    # Get geodesic parameters from the two points
    p = Geodesic.WGS84.Inverse(P[0], P[1], Q[0], Q[1])
    # Distance between the two points (m)
    D = p["s12"]
    # Azimuth of the line at point P
    azi = p["azi1"]
    # Get the coordinates of N points along the geodesic line
    points = [(P[0], P[1])]
    # Create line with starting point P and the previously computed azimuth
    l = Geodesic.WGS84.Line(P[0], P[1], azi)
    for i in range(1, N - 1):
        # Compute the lat, lon position of the ith point
        o = l.Position(i * D / (N - 1))
        points.append((o["lat2"], o["lon2"]))
    points.append((Q[0], Q[1]))  # add the last point
    return np.array(points)


def geodesic_line_distance(P: Tuple[float], Q: Tuple[float], N: int):
    """
    Returns the angle and distances of the geodesic line formed of N points between points P and Q.
    P : (lat, long)
    Q : (lat, long)
    N : number of points

    returns : azimuth as seen from P, np.array formed of the N (distance from P) points
    """
    # Get geodesic parameters from the two points
    p = Geodesic.WGS84.Inverse(P[0], P[1], Q[0], Q[1])
    # Distance between the two points (m)
    D = p["s12"]
    # Azimuth of the line at point P
    azi = p["azi1"]
    return azi, np.linspace(0, D, N)


class Cache:
    """
    Cache xarray datasets using a cache string which describes the Dataset.

    By default, processed files are cached in ~/.cache/SSW-2D, this can be
    overriden by setting the SSW_CACHE_PATH environment variable.
    To disable caching, the environment variable SSW_DISABLE_CACHE has to be
    set to 1

    Every time a dataset is cached, it's hash and cache_string are stored
    in the cache.index file.

    TODO: Make a clean() function
    """

    def __init__(self, cache_string: str):
        """
        Cache a dataset with it's cache string .

        cache_string : string that describes the content of the dataset (should be on a single line)
        """
        self.cache_string = cache_string
        path = getenv("SSW_CACHE_PATH")
        if path == None:
            self.path = user_cache_dir("SSW-2D")
        else:
            self.path = path
        disable_cache = getenv("SSW_DISABLE_CACHE")
        if bool(disable_cache):
            self.cache_enabled = False
        else:
            self.cache_enabled = True
        # Create cache directory if necessary
        makedirs(self.path, exist_ok=True)
        # Hash the cache string with sha256, keep the 64th first characters
        self.hash = hashlib.sha256(self.cache_string.encode("utf-8")).hexdigest()[0:64]
        # File path
        self.fp = join(self.path, f"{self.hash}.nc")

    def has(self) -> bool:
        """
        Checks if the given dataset exists in the cache
        """
        return isfile(self.fp) and self.cache_enabled

    def store(self, ds: xr.Dataset):
        """
        Store the dataset in the cache if not already stored
        """
        if self.cache_enabled and not self.has():
            # Store file
            ds.to_netcdf(self.fp)
            # Write hash and cache string to cache index
            with open(join(self.path, "cache.index"), "a") as f:
                f.write(f"{self.hash}\t{self.cache_string}\n")

    def get(self) -> xr.Dataset:
        """
        Retrieve the dataset from the cache if it exists
        """
        if self.cache_enabled and self.has():
            return xr.load_dataset(self.fp)

    def clear(self):
        """
        Remove everything from the cache
        """
        rmdir(self.path)
