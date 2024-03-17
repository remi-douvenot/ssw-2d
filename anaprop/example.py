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
# @package anaprop/example.py
# @author Remi Douvenot - storca -
# @brief Plotting of interpolated and calculated refraction index along a given (P, Q) axis
# @warning
##

from atmosphere import load, add_n, add_grad_n, interpolate_vertical
from relief import get_ign_height_profile
import matplotlib.pyplot as plt

profile_color = "black" # color of the height profile plot
path = "/media/storca/COMMUN/stage_ssw/ZTU_20130707.grb_jpeg" # path of the initial dataset
P = (43.957073, 1.402456) # First point of the axis
Q = (43.576164, 6.992977) # Second point of the axis
N = 500 # Number of interpolated data points allong the (P, Q) axis and along the (0, hmax) alt range
# Load the initial dataset and interpolate it between P and Q, drop unused data
i = load(path, P, Q, N)
# Retrieve the height profile from IGN 
h = get_ign_height_profile(P, Q, N)
# Compute the refraction index
i = add_n(i)
# Compute the gradient of the refraction index
i = add_grad_n(i)
# Interpolate all the dataset in between pressure levels, and convert pressure levels to meters
i = interpolate_vertical(i, 500)
# Remove the data that is below the height profile
i = i.where(i.height > h.height)

# Create new figure
plt.figure()
i.t.sel(step="02:00:00").plot() # plot temperature at 2h
h.height.plot(color=profile_color) # plot the profile
# Repeat for other variables
plt.figure()
i.r.sel(step="02:00:00").plot()
h.height.plot(color=profile_color)
plt.figure()
i.grad_n.sel(step="02:00:00").plot()
h.height.plot(color=profile_color)
# Show the figures
plt.show()