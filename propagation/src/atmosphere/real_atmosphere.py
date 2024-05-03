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
# @author Remi Douvenot - storca -
# @brief Read data from atmospherical datasets, compute refraction indexes and interpolate the data
# @warning
##

import xarray as xr
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import basename

from src.classes_and_files.classes import Cache
import src.classes_and_files.geodesic as geo
import src.atmosphere.cds as cds

def normalize(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize the variable names among all datasets
    """
    # variables mappings old=new
    vm = dict(
        pres="isobaricInhPa",
    )
    # coordinates mappings old=new
    cm = dict(
        heightAboveGround="heights" #FIXME : heightAboveGround is NOT height above the sea!
    )
    coords = list(ds.coords)
    variables = list(ds.keys())

    for c in coords:
        if c in cm.keys():
            # Rename coordinates
            ds = ds.rename({c:cm[c]})
            #ds = ds.assign_coords({cm[c]:ds[c]})
    for v in variables:
        if v in vm.keys():
            # Rename variables
            ds = ds.rename({v:vm[v]})
    return ds


def load(path: str, P: Tuple[float], Q: Tuple[float], N: int) -> xr.Dataset:
    """
    Loads a given dataset, interpolates it on a line of N values between points P and Q and returns a smaller dataset.

    path : path to a grib/netcdf dataset
    P : latitude, longitude tuple of the first point
    Q : latitude, longitude tuple of the second point
    N : number of points to interpolate
    ds : xarray dataset
    """
    # Check for cached datasets
    cache_string = f"atmosphere,{P},{Q},{N},{basename(path)}"
    c = Cache(cache_string)
    ds = None
    if c.has():
        print("[*] Using cached atmosphere slice")
        ds = c.get()
    else:
        # Load the initial dataset
        print("[*] Loading initial atmosphere dataset...")
        ds = xr.load_dataset(path, engine="cfgrib")

        print("[*] Interpolating positions...")
        # (P, Q) line (lat, lon) coordinates
        lc = geo.geodesic_line_coords(P, Q, N)
        # (P, Q) line; azi : angle from P, ld = distances from P
        azi, ld = geo.geodesic_line_distance(P, Q, N)

        # Warning : brain juice below
        # Interpolate on the (P,Q) segment, using (lat, lon) couples by creating a new coordinate d which is the distance from P on the axis (P,Q)
        ds = ds.interp(
            coords=dict(latitude=("d", lc[:, 0]), longitude=("d", lc[:, 1])),
            method="linear",
        )
        # Add the distances array to the new coordinate
        ds = ds.assign_coords(dict(d=ld))

        # Add metadata
        ds["d"] = ds.d.assign_attrs({"long_name": "Distance from P", "units": "m"})

        # Cache dataset
        c.store(ds)
    return ds


def height_to_hPa(height, P0=101325, T=288.15):
    M = 29e-3  # molar mass of air kg/mol
    g = 9.80665  # standard gravity (m/s^2)
    k = 1.38e-23  # boltzman constant
    Na = 6.02e23  # avogadro number
    return P0 * np.exp(-M * g * height / (k * T * Na)) / 100


def add_n(ds: xr.Dataset):
    def N(x: xr.Dataset):
        """
        Given the dataset, returns the set of operations required to compute a new variable
        """
        # Computes the refractive index based on atmospherical data
        # Formulas and constants taken from ITU-R P.453-14
        a = 6.1121
        b = 18.678
        c = 257.14
        d = 234.5

        # Rename and convert dataset variables
        P_hPa = x.isobaricInhPa
        T_degK = x.t
        T_degC = x.t - 273.15 # Convert K to °C
        H_percent = x.r # relative humidity in %

        EF_water = 1 + 1e-4 * (7.2 + P_hPa * (0.0320 + 5.9 * 1e-6 * T_degC * T_degC))
        es = EF_water * a * np.exp(((b - T_degC / d) * T_degC )/ (T_degC + c))

        e = H_percent * es / 100

        N = 77.6 * P_hPa / T_degK - 5.6 * e / T_degK + 3.75e5 * e / (T_degK * T_degK)
        return N

    print("[*] Adding refractive indexes...")
    # Create a new varaible N with function N(ds)
    ds = ds.assign(N=N)
    ds["N"] = ds.N.assign_attrs({"long_name": "Refractivity"}) # add metadata
    # Create a new variable 'n' using conversion between N and n
    ds = ds.assign(n=lambda ds: 1 + ds.N * 1e-6)
    ds["n"] = ds.n.assign_attrs({"long_name": "Refractive index"}) # add metadata
    return ds

def add_M(ds: xr.Dataset) -> xr.Dataset:
    Re = 6.371e6 # earth radius (meters)
    # M = (h/Re * 1e6) + N
    def M(ds:xr.Dataset):
        return ds.heights/Re * 1e6 + ds.N
    ds = ds.assign(M=M)
    ds["M"] = ds.M.assign_attrs({"long_name": "Corefractivity index"})
    return ds

def add_grad_m(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign(grad_m=lambda x: x.m.differentiate("heights"))
    ds["grad_m"] = ds.grad_m.assign_attrs(
        {"long_name": "Co-refractive index gradient", "units": "m⁻¹"}
    )
    return ds

def add_grad_n(ds: xr.Dataset):
    """
    Compute de gradient of the refraction index along the vertical
    """
    print("[*] Computing gradients of refractive indexes...")
    # Create a new variable grad_n using function "DataArray.differentiate"
    ds = ds.assign(grad_n=lambda x: x.n.differentiate("heights"))
    # Add metadata for plotting
    ds["grad_n"] = ds.grad_n.assign_attrs(
        {"long_name": "Refraction index gradient", "units": "m⁻¹"}
    )
    # Differentiate N along height
    ds = ds.assign(grad_N=lambda ds: ds.N.differentiate("heights"))
    ds = ds.assign(grad_N=lambda ds: ds.grad_N * 1000)
    ds["grad_N"] = ds.grad_N.assign_attrs(
        {"long_name": "Refractivity gradient", "units": "km⁻¹"}
    )
    return ds

def add_heights(ds: xr.Dataset, hmax=3000):
    """
    Replace the isobaricInhPa coordinate with heights if necessary
    """
    print("[*] Adding heights...")
    coords = list(ds.coords)
    heights = np.linspace(0, hmax, len(ds.isobaricInhPa), dtype=float)
    # Create the corresponding pressure array
    pressures = height_to_hPa(heights, 1e5)
    if 'isobaricInhPa' in coords and not 'heights' in coords:
        ds = ds.interp(coords=dict(isobaricInhPa=("heights", pressures)), method="slinear")
        # Assign the new coordinate to the generated heights
        ds = ds.assign_coords({"heights": heights})
    return ds

def interpolate_vertical(ds: xr.Dataset, N, hmax=2500):
    """
    Interpolate the dataset in between pressure levels and
    convert the pressure levels to meters
    """
    print(f"[*] Interpolating heights... N={N}, hmax={hmax}")
    # Create an array of N heights in meters 
    heights = np.linspace(0, hmax, N, dtype=float)
    print(heights[0], heights[-1])
    coords = list(ds.coords)
    if 'heights' in coords:
        ds = ds.interp(heights=heights, method="linear")
        return ds
    elif 'isobaricInhPa' in coords:
        print('isobaricinhpa')
        # Create the corresponding pressure array
        pressures = height_to_hPa(heights, 1.013e5)
        # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial'
        # Interpolate and replace the isobaricInhPa coord by height
        ds = ds.interp(coords=dict(isobaricInhPa=("heights", pressures)), method="linear")
        # Assign the new coordinate to the generated heights
        ds = ds.assign_coords({"heights": heights})
        # Edit the existing dataset metadata in place
        ds["heights"] = ds.heights.assign_attrs({"long_name": "Height", "units": "m"})
    return ds

def get_corefractive_index(config) -> np.ndarray:
    """
    Required parameters in config

    model : atmospherical data to use, ERA5 only for now
    when : datetime
    P : (lat,lon) tuple
    Q : (lat,lon) tuple
    N_x : sampling over x
    N_z : sampling over z
    z_step : difference in height between two samples

    Returns the corefractive matrix index m of size (N_x * N_z) given the inputs
    """
    c = config # rename config

    data_path = os.getenv('SSW_DATA_PATH')
    if data_path == None:
        raise ValueError(f"The SSW_DATA_PATH environment variable must be set.")

    if c.atmosphere == "ERA5":
        path = cds.download_dataset(c.atmosphere_datetime, data_path)
    else:
        raise ValueError(f"Unsupported weather source \"{c.atmosphere}\"")
    
    # Compute maximum height
    hmax = c.N_z * c.z_step

    # Create time string used to select data
    time = c.atmosphere_datetime.strftime("%Y-%m-%dT%H:00:00.%f")
    # Load initial atmosphere dataset
    a = load(path, c.P, c.Q, c.N_x)
    # Normalize vars and coords names
    a = normalize(a)
    # Compute the refraction index
    a = add_n(a)
    a = add_heights(a, hmax)
    a = add_M(a)
    # Add m
    a = a.assign(m = lambda ds: ds.n + ds.heights/6.371e6)
    # Compute the gradient of the refraction index
    a = add_grad_n(a)
    a = add_grad_m(a)
    # Interpolate all the dataset in between pressure levels, and convert pressure levels to meters
    a = interpolate_vertical(a, c.N_z, hmax)
    # Save the gradient into a file to display it in the UI
    np.save('./inputs/refractivity_gradient.npy', a.grad_m.sel(time=time).to_numpy().T) # save to show it in the UI later
    # transpose the array, shape of (N_x, N_z) instead of (N_z, N_x)
    return a.m.sel(time=time).to_numpy().T


if __name__ == "__main__":
    # Unused test code
    print("Reading weather data... ", end="")
    print("ok!")
    # l = common.geodesic_line_coords((43.957073, 1.402446), (43.566163, 6.992977), 10)
    # azi, d = common.geodesic_line_distance((43.957073, 1.402446), (43.566163, 6.992977), 10)
    # interpolate_2D((43.957073, 1.402446), (49.568123, 15.059957), 100, ds)
    # ds.t.sel(step="02:00:00", isobaricInhPa=950.0).plot()
    # plt.show()
    path = "/media/storca/COMMUN/stage_ssw/ZTU_20130707.grb_jpeg"
    i = load(path, (43.957073, 1.402456), (43.576164, 6.992977), 500)
    i = add_n(i)
    print(i)
    plt.figure()
    i.t.sel(step="02:00:00").plot()
    plt.figure()
    i.r.sel(step="02:00:00").plot()
    plt.figure()
    i.n.sel(step="02:00:00").plot()
    plt.show()