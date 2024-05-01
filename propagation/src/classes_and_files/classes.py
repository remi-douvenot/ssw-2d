# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

import os
from os.path import isfile, join
from platformdirs import user_cache_dir
import hashlib
import xarray as xr

# ------------------------ #
# --- Defining classes --- #
# ------------------------ #
class Config:
    def __init__(self):
        self.N_x = 0
        self.N_y = 0
        self.N_z = 0
        self.x_step = 0
        self.y_step = 0
        self.z_step = 0
        self.x_s = 0 # distance of the source (negative value)
        self.freq = 0
        self.max_compression_err = 0 # Max compression error
        self.V_s = 0 # compression threshold on signal
        self.V_p = 0 # compression threshold on propagator
        self.wv_family = 'None'
        # wavelet level
        self.wv_L = 0
        # type of apodisation window
        self.apo_window = 'None'
        # percentage of apodisation of the domain along y
        self.apo_y = 0
        # percentage of apodisation of the domain along z
        self.apo_z = 0
        # percentage of image layer of the domain along z (if any ground)
        self.image_layer = 0
        # number of point sin the image layer (multiple of 2^L)
        self.N_im = 0
        # ground type ('None', 'PEC', or 'dielectric')
        self.ground = 'None'
        # ground relative permittivity (for dielectric ground only)
        self.epsr = 0
        # ground conductivity (for dielectric ground only)
        self.sigma = 0
        self.atmosphere = 'None'  # atmospheric profile type
        self.c0 = 0  # standard atm gradient
        self.delta = 0  # evaporation duct height
        self.zb = 0  # base height of a trilinear duct
        self.c2 = 0  # gradient in a trilinear duct
        self.zt = 0  # thickness of a trilinear duct
        self.atm_filename = 'None'  # file for a hand-generated atmospheric profile
        self.turbulence = 'N'
        self.Cn2 = 0
        self.L0 = 0
        self.py_or_cy = 'None'  # ='Py' for Python code, and = 'Cy' for Cython code
        self.case_index = None
        self.atmosphere_datetime = None
        self.P = None
        self.Q = None


class Cache:
    """
    Cache xarray datasets using a cache string which describes the Dataset.

    By default, processed files are cached in ~/.cache/SSW-2D, this can be
    overriden by setting the SSW_CACHE_PATH environment variable.
    To disable caching, the environment variable SSW_DISABLE_CACHE has to be
    set to 1
    Cache is cleaned at each initialization, it keeps the last accessed 
    SSW_CACHE_NB files (20 by default), and removes the other files. 

    Every time a dataset is cached, it's hash and cache_string are stored
    in the cache.index file.

    TODO: Make a clean() function
    """

    def __init__(self, cache_string: str):
        """
        Cache a dataset with it's cache string.

        cache_string : string that describes the content of the dataset (should be on a single line)
        """
        self.cache_string = cache_string
        # get SSW_CACHE_PATH environment variable
        path = os.getenv("SSW_CACHE_PATH")
        if path == None:
            self.path = user_cache_dir("SSW-2D")
        else:
            self.path = path
        # get SSW_DISABLE_CACHE environment variable
        disable_cache = os.getenv("SSW_DISABLE_CACHE")
        if bool(disable_cache):
            self.cache_enabled = False
        else:
            self.cache_enabled = True
        # get SSW_CACHE_SIZE environment variable
        cache_size = os.getenv("SSW_CACHE_SIZE")
        if cache_size == None:
            self.cache_size = 20 
        else:
            self.cache_size = int(cache_size)
        self.cache_size += 1 # cache.index file does not count 
        # Create cache directory if necessary
        os.makedirs(self.path, exist_ok=True)
        # Hash the cache string with sha256, keep the 64th first characters
        self.hash = hashlib.sha256(self.cache_string.encode("utf-8")).hexdigest()[0:64]
        # File path
        self.fp = join(self.path, f"{self.hash}.nc")
        # Clean the cache
        self.clean()

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
        os.rmdir(self.path)

    def clean(self):
        """
        Remove unused files in the cache

        Keeps the last accessed SSW_CACHE_NB elements (20 by default), removes the other ones.
        """
        # Get files names and last accessed time
        cache_files = {}
        for root, dirs, filenames in os.walk(self.path):
            for fn in filenames:
                s = os.stat(os.path.join(self.path, fn))
                # st_atime gives the unix time in seconds when file was last accessed
                cache_files[s.st_atime] = fn
            break   #prevent descending into subfolders
        
        # Remove the files that should not be here anymore
        keys = sorted(cache_files)
        # If there are more files than there should be
        if len(keys) > self.cache_size:
            for i in range(len(keys) - self.cache_size):
                fp = os.path.join(self.path, cache_files[keys[i]])
                os.remove(fp)

# ---------- END --------- #
# --- Defining classes --- #
# ------------------------ #
config = Config() # instanciate config object