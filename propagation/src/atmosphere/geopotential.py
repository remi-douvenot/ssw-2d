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
# @author storca
# @brief Regrid datasets from isobaric levels into geopotential height levels
# @warning
##

import numpy as np
from scipy.interpolate import CubicSpline
import xarray as xr

def regrid_dataset(ds: xr.Dataset, N_z, z_max) -> xr.Dataset:
    """
    Original dataset contains data in terms of isobaric surfaces.
    This function re-grids the dataset in terms of geopotential height.

    ds : xarray dataset
    N_z : vertical sampling
    z_max : maximum altitude (meters)

    returns : the re-gridded dataset
    """
    # Dirty, but too many variables to pass to other functions
    global d, P, time, h, Z

    # Old coordinates
    d = ds.d.data
    P = ds.isobaricInhPa.data
    time = ds.time.data

    # New coordinate
    h = np.linspace(0, z_max, N_z)

    # Variables
    r = ds.r.data
    t = ds.t.data
    z = ds.z.data

    # Geopotential atlitudes
    Z = ds.z.data/9.81

    # New variables
    new_shape = (len(time), N_z, len(d))
    new_P = np.zeros(new_shape, dtype=float)
    new_r = np.zeros(new_shape, dtype=float)
    new_t = np.zeros(new_shape, dtype=float)
    new_z = np.zeros(new_shape, dtype=float)

    # Call regriding functions
    regrid_P(new_P)
    regrid_data(r, new_r)
    regrid_data(t, new_t)
    regrid_data(z, new_z)

    # Instanciate the new dataset
    new_ds = xr.Dataset(
        data_vars=dict(
            r=(['time', 'heights', 'd'], new_r),
            t=(['time', 'heights', 'd'], new_t),
            z=(['time', 'heights', 'd'], new_z),
            isobaricInhPa=(['time', 'heights', 'd'], new_P),
        ),
        coords=dict(
            time=time,
            heights=h,
            d=d
        ),
        attrs=ds.attrs
    )
    # Add metadata
    new_ds["heights"] = new_ds.heights.assign_attrs({"long_name": "Geopotential height", "units": "m"})
    new_ds["d"] = new_ds.d.assign_attrs(ds.d.attrs)
    new_ds["r"] = new_ds.r.assign_attrs(ds.r.attrs)
    new_ds["t"] = new_ds.t.assign_attrs(ds.t.attrs)
    new_ds["z"] = new_ds.z.assign_attrs(ds.z.attrs)
    new_ds["isobaricInhPa"] = new_ds.isobaricInhPa.assign_attrs(ds.isobaricInhPa.attrs)
    return new_ds

def regrid_P(new_P, scipy_interp_func=CubicSpline):
    """
    Regrid pressure data
    """
    for time_i in range(len(time)):
        for d_i in range(len(d)):
            # Local heights corresponding to each pressure level
            z_local = Z[time_i,:,d_i]
            # Coordinates of the pressure in terms of geopotential height
            points = z_local
            # Values of pressure at each height
            values = P
            P_func = np.vectorize(scipy_interp_func(points, values))

            new_P[time_i, :, d_i] = P_func(h)

def regrid_data(data, new_data, scipy_interp_func=CubicSpline):
    """
    Regrid dataset variable
    """
    for time_i in range(len(time)):
        for d_i in range(len(d)):
            # Local heights corresponding to each pressure level
            z_local = Z[time_i,:,d_i]
            # Coordinates of the data in terms of geopotential height
            points = z_local
            # Values of the given data
            values = data[time_i,:,d_i]
            D_func = np.vectorize(scipy_interp_func(points, values))

            new_data[time_i, :, d_i] = D_func(h)
