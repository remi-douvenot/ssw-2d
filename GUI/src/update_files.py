# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

import csv
import pandas as pd


# --- write config file --- #
def update_file(element, value, filename):
    # print('Write', input, 'in', filename, 'file !!')
    if filename == 'propa':
        file_loc = '../propagation/inputs/configuration.csv'
    elif filename == 'source':
        file_loc = '../source/inputs/configuration.csv'
    elif filename == 'terrain':
        file_loc = '../terrain/inputs/conf_terrain.csv'
    else:
        raise ValueError('Input file is not valid in update_files.py')

    file_to_update = open(file_loc)
    # reading the csv file as a dataframe
    dataframe = pd.read_csv(file_to_update)
    # create a Series with it
    series_file = pd.Series(data=dataframe.iloc[:, 1].values, index=dataframe.iloc[:, 0].values)
    # uu = df.loc[(df[0] == input)]  # = [[input, value]]

    # update the value in the series_file
    series_file.loc[element] = value

    # back from series to dataframe
    # print(series_file.values)
    dataframe.iloc[:, 1] = series_file.values
    # writing into the file
    dataframe.to_csv(file_loc, index=False)

    # print('Write', input, 'in', filename, 'file !!')


