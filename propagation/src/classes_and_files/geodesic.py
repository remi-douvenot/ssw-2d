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
# @brief Geodesic computations for SSW-2D
# @warning
##

from geographiclib.geodesic import Geodesic
import numpy as np
from typing import Tuple

def geodesic_line_coords(P: Tuple[float], Q: Tuple[float], N: int) -> np.array:
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
