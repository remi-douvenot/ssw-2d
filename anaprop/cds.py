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
# @package anaprop/cds.py
# @author Remi Douvenot - storca -
# @brief Download ERA5 reanalysis atmospherical datasets from the EU Copernicus datastore
# @warning Copernicus user ID and API key should be setup
##

import cdsapi
from datetime import datetime
import os
from platformdirs import user_cache_dir
import shutil

def download_dataset(d: datetime, save_dir='./') -> str:
    """
    Download a GRIB dataset from the EU Copernicus DataStore.
    grid : 0.25° both on longitude and latitude

    d: datetime, desired time of observation (time in UTC, Zulu timezone)

    returns : path to the downloaded dataset
    """
    # Initialize file name, path and make directories
    fn = f'era5-{d.strftime("%Y-%m-%d-%HH")}.grib'
    temp_path = os.path.join(user_cache_dir('SSW2D-downloads'), fn)
    data_path = os.path.join(save_dir, fn)
    os.makedirs(user_cache_dir('SSW2D-downloads'), exist_ok=True)

    # Check if a previously downloaded file is present
    if not(os.path.isfile(data_path)):
        if not os.path.isfile(temp_path):
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-pressure-levels', # dataset name
                {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': [
                        'relative_humidity', 'temperature', 'geopotential' # parameters to download
                    ],
                    'pressure_level': [ # pressure levels to download
                        '125', '150', '175',
                        '200', '225', '250',
                        '300', '350', '400',
                        '450', '500', '550',
                        '600', '650', '700',
                        '750', '775', '800',
                        '825', '850', '875',
                        '900', '925', '950',
                        '975', '1000',
                    ],
                    'area': [ # region in which data should be
                        53, -8, 38,
                        12,
                    ],
                    'year': f'{d.year}',
                    'month': f'{d.month:02d}', #month with 2 digits (leading zeroes)
                    'day': f'{d.day:02d}', #idem
                    'time': [
                        f'{d.hour-1:02d}:00', f'{d.hour:02d}:00', f'{d.hour+1:02d}:00',
                    ], #since a download takes a LOT of time, take also the previous and the next hour
                },
            temp_path)

        # Move the file from it's temporary location to it's desired location
        # NOTE : if an error occurs here, the file will still be cached
        shutil.move(temp_path, data_path)
    return data_path

if __name__ == '__main__':
    d = datetime(2016, 8, 13, 7) # Montdidier
    download_dataset(d, '/media/storca/COMMUN/stage_ssw/')