#!/usr/bin/python3
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
# @package anaprop/cases.py
# @author Remi Douvenot - storca -
# @brief Download, proccess and plot any anaprop case listed in a csv file
# @warning
##

from datetime import datetime
from typing import List, Dict
import sys

import matplotlib.pyplot as plt
import xarray as xr

import atmosphere as atm
import relief as rel
import cds

profile_color = "black" # color of the height profile plot

class Case:
    """
    Describes a single anaprop case
    """
    def __init__(self, name:str, when:str, P:str, Q:str, f:str, alt:str):
        """
        Initialize a case
        name : Case name (str)
        when : string of datetime, passed into datetime ctor
        P : "lat,lon" string of the first point
        Q : "lat,lon" string of the second point
        """
        self.name = name
        self.when = datetime.fromisoformat(when)
        self.P = tuple(float(e) for e in P.split(','))
        self.Q = tuple(float(e) for e in Q.split(','))
        self.f = float(f)
        if alt == "QFE":
            self.alt = 0.0
        else:
            self.alt = float(alt)
    def __str__(self):
        return f'<Case {self.name} on the {self.when.strftime("%d/%m/%Y")} at {self.when.strftime("%H")}H>'


def load(path='./cases.csv') -> List[Case]:
    """
    Load cases from csv file
    """
    cases = []
    with open(path, 'r') as f:
        f.readline() #skip headers
        for line in f:
            c = line.rstrip().split("\t")
            cases.append(
                Case(
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    c[4],
                    c[5]
                )
            )
    return cases

def show(cases:List[Case]):
    """
    Nicely display different cases
    """
    for i, case in enumerate(cases):
        print(f"{i} : {case}")

def replay(case:Case, data_path='./', N=300, weather_source="era5"):
    if weather_source == "era5":
        path = cds.download_dataset(case.when, data_path)
    else:
        raise ValueError(f"Unsupported weather source \"{weather_source}\"")
    # Load initial dataset
    a = atm.load(path, case.P, case.Q, N)
    # Normalize vars and coords names
    a = atm.normalize(a)
    # Retrieve the height profile from IGN 
    h = rel.get_ign_height_profile(case.P, case.Q, N)
    # Compute the refraction index
    a = atm.add_n(a)
    a = atm.add_heights(a)
    a = atm.add_M(a)
    # Compute the gradient of the refraction index
    a = atm.add_grad_n(a)
    a = atm.add_grad_M(a)
    # Interpolate all the dataset in between pressure levels, and convert pressure levels to meters
    a = atm.interpolate_vertical(a, N)
    # Remove the data that is below the height profile
    a = a.where(a.heights > h.height)

    time = case.when.strftime("%Y-%m-%dT%H:00:00.%f") # time format like this : "2019-02-27T16:00:00.000000000"

    # Create new figure
    plt.figure()
    # plot temperature at time
    a.t.sel(time=time).plot()
    h.height.plot(color=profile_color) # plot the profile
    # Repeat for other variables
    plt.figure()
    #i.r.sel(step="02:00:00").plot()
    a.M.sel(time=time).plot()
    h.height.plot(color=profile_color)
    plt.figure()
    a.grad_M.sel(time=time).plot()
    h.height.plot(color=profile_color)
    # Show the figures
    plt.title(f"{weather_source.upper()} - {case.name}")
    plt.figure()
    xr.plot.contour(a.grad_M.sel(time=time), levels=1)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        cases = load("cases.csv")
        if n < len(cases):
            # show(cases)
            s = cases[n]
            replay(s, '/media/storca/COMMUN/stage_ssw')