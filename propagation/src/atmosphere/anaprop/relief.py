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
from typing import Tuple, List
import numpy as np
#import common
import src.atmosphere.anaprop.common as common
from colorama import Fore
import matplotlib.pyplot as plt

import os

def encode_points(points:List[Tuple[float]]) -> str:
    """
    Bing maps points compression algorithm. Tested, works.

    Translated from
    https://learn.microsoft.com/en-us/bingmaps/rest-services/elevations/point-compression-algorithm#javascript-implementation
    """
    latitude = 0
    longitude = 0
    result = []
  
    for point in points:
        # step 2
        new_latitude = round(point[0] * 100000)
        new_longitude = round(point[1] * 100000)
  
        # step 3
        dy = new_latitude - latitude
        dx = new_longitude - longitude
        latitude = new_latitude
        longitude = new_longitude
  
        # step 4 and 5
        dy = (dy << 1) ^ (dy >> 31 if dy < 0 else 0)
        dx = (dx << 1) ^ (dx >> 31 if dx < 0 else 0)
  
        # step 6
        index = ((dy + dx) * (dy + dx + 1) // 2) + dy
  
        while index > 0:
            # step 7
            rem = index & 31
            index = (index - rem) // 32
  
            # step 8
            if index > 0:
                rem += 32
  
            # step 9
            result.append("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"[rem])
  
    # step 10
    return "".join(result)

def request_bing_points(points: List[Tuple[float]], key:str, heights="ellipsoid") -> List:
    """
    Request a list of point heights from bing api

    points : list of (lat, lon) max 1024 points
    key : Bing maps api key
    heights : "elispoid" or "sealevel"
    """
    # API Docs : https://learn.microsoft.com/en-us/bingmaps/rest-services/elevations/get-elevations

    # Convert (lat, lon) list to lat1,lon1,lat2,lon2,...,latN,lonN string list
    # points = [str(p[i]) for p in points for i in range(2)]
    # points = ",".join(points) # convert points list to str
    # Pack the arguments into the url
    url = f"http://dev.virtualearth.net/REST/v1/Elevation/List?points={encode_points(points)}&heights={heights}&key={key}"
    response = r.get(url) # Call the API
    if response.status_code == 200:
        d = response.json()
        elevations = d['resourceSets'][0]['resources'][0]['elevations'] # access elevations in the json
        return [float(e) for e in elevations] # convert elevations to float
    else:
        raise IOError(f"Bing API error : HTTP{response.status_code}\n{response.content}")


def get_bing_height_profile(P: Tuple[float], Q: Tuple[float], N: int, model="ellipsoid") -> xr.Dataset:
    """
    Retrieve height profile using bing maps API.

    P : (lat, lon) first point
    Q : (lat, lon) second point
    N : sampling
    model : earth model used : "elispoid" -> WGS84 or "sealevel" -> EGM2008 2.5’

    IMPORTANT : BING_MAPS_API_KEY environment variable must be set, head to https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key to obtain a key
    """
    key = os.getenv('BING_MAPS_API_KEY')
    if key == None:
        raise ValueError("To download height profiles from Bing, BING_MAPS_API_KEY environment variable must be set, head to https://www.bingmapsportal.com/ to get one.")

    # Cache string made with P, Q and N
    cache_string = f"bing_relief,{P}{Q}{N}{model}"
    c = common.Cache(cache_string) # instanciate object
    if c.has(): # if a match exists in the cache
        print(f"[*] Using cached profile")
        return c.get() # return the cached dataset
    
    # A maximum of 1024 elevations can be requested at once, we have to split the requests
    coords = common.geodesic_line_coords(P, Q, N)

    # Get the azimuth at P and the distance bewteen P and Q in meters 
    azi, distances = common.geodesic_line_distance(P, Q, N)

    heights = []
    # Process whole chunks of 1024 points
    for i in range(N//1024):
        heights += request_bing_points(coords[i*1024:(i+1)*1024], key, model)

    # Process the remaining ones
    start = N//1024 * 1024
    heights += request_bing_points(coords[start:], key, model)
    
    # Create the dataset
    ds = xr.Dataset(
        data_vars=dict(
            height=(["d"], heights),
        ),
        coords=dict(d=("d", distances)),
        attrs={
            "description": "Height profile from P to Q, with N points, downloaded from Microsoft Bing Maps",
            "P": P,
            "Q": Q,
            "azi": azi,
            "SSW_TYPE": "relief",
        },
    )
    c.store(ds)  # store the new dataset in the cache
    return ds

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
        print(d)
        # Initialize data holders
        h = np.zeros(N, dtype=float)
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
        print(os.getcwd())
        return ds
    else:
        print("Error:", response.status_code)
        print(response.content)


def plot(ds: xr.Dataset):
    ds.height.plot()
    plt.show()

def compare():
    # Cas lézigan-corbières - chambéry
    P = (43.173414, 2.738417)
    Q = (45.638015, 5.882924)
    N = 1029
    ds_bing = get_bing_height_profile(P, Q, N)
    ds_bing.height.plot(label="Bing data")
    ds_ign = get_ign_height_profile(P, Q, N)
    ds_ign.height.plot(label="IGN data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare()
    # print(ds)
    # points = common.geodesic_line_coords((43.957073, 1.402446), (43.566163, 6.992977), 4)
    # print(",".join([str(p[i]) for p in points for i in range(2)]))
    # print(encode_points(points))