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
# @package anaprop/relief.py
# @author Remi Douvenot - storca -
# @brief Retrieve 2D height profiles from IGN
# @warning
##

import requests as r
import xarray as xr
import json
from typing import Tuple
import numpy as np
import common
from colorama import Fore
import matplotlib.pyplot as plt


def get_ign_height_profile(P: Tuple[float], Q: Tuple[float], N: int) -> xr.Dataset:
    if N > 5000:
        raise ValueError(
            "N must be smaller than 5000\ncf https://geoservices.ign.fr/documentation/services/services-geoplateforme/altimetrie#72673"
        )

    # Ressource url
    url = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevationLine.json"
    # HTTP headers
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    # Request parameters
    data = {
        "lat": f"{P[0]},{Q[0]}",
        "lon": f"{P[1]},{Q[1]}",
        "resource": "ign_rge_alti_wld",
        "delimiter": ",",
        "indent": "false",
        "measures": "false",
        "zonly": "true",
        "sampling": f"{N}",
    }

    # Cache string made with the request data
    cache_string = f"relief,{json.dumps(data)}"
    c = common.Cache(cache_string)
    if c.has():
        print(f"[*] Using cached profile")
        return c.get()

    print("[*] Downloading height profile...")
    response = r.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # Convert JSON response to dict()
        d = response.json()
        # Initialize data holders
        h = np.zeros(n, dtype=float)
        heights = []
        # Store the heights of the profile in the array
        for i in range(len(d["elevations"])):
            h[i] = float(d["elevations"][i]["z"])

        # Get the azimuth at P and the distance bewteen P and Q in meters 
        azi, distances = common.geodesic_line_distance(P, Q, N)

        # Create the dataset
        ds = xr.Dataset(
            data_vars=dict(
                height=(["d"], h[:N]),  # BUG: the API sometimes returns N+1 heights
            ),
            coords=dict(d=("d", distances)),
            attrs={
                "description": "Height profile from P to Q, with N points, downloaded from IGN elevationLine.json using ign_rge_alti_wld",
                "P": P,
                "Q": Q,
                "azi": azi,
                "url": url,
                "SSW_TYPE": "relief",
                "request_json": json.dumps(data),
            },
        )
        c.store(ds)  # store the new dataset in the cache
        return ds
    else:
        print("Error:", response.status_code)
        print(response.content)


def plot(ds: xr.Dataset):
    ds.height.plot()
    plt.show()


if __name__ == "__main__":
    ds = get_ign_height_profile((43.957073, 1.402446), (43.566163, 6.992977), 5000)
    plot(ds)
    print(ds)
