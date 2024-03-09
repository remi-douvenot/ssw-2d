Date: 05 April 2022
Last modif.: 10 Nov. 2022
Author: Rémi Douvenot

Table of Content

I/ Licence
II/ Installation and dependencies
III/ Documentation
IV/ Contact
V/ How to cite


I/ LICENCE

SSW-2D is a free and open software is under the GNU General Public License.
See COPYING.txt for details


II/ INSTALLATION

SSW-2D requires python3 and the following 
packages to run the core propagation module:
- numpy
- scipy
- pywavelets
- matplolib
- cython

and theses additional packages to run the GUI:
- PyQT5
- Pandas

# ---------------------------------------------#
# ------------- LINUX -------------------------#
# ---------------------------------------------#

Installation detailled procedure:

install snap
>> sudo apt install snap

install Pycharm (adapt installation to your favorite IDE)
>> sudo snap install pycharm-community

install pip3 (to install python packages)
>> sudo apt install python3-pip

install required packages (user scope)
>> pip3 install -r requirements.txt

install additional required packages for using the GUI (user-scope)
>> pip3 PyQt5 pandas

to modify the GUI, install designer
>> sudo apt install qttools5-dev-tools 
cmd to launch the GUI designer is >> designer

to load weather data, additionnal depencencies are required :
>> pip3 install xarray cfgrib
along with the eccodes system library (required by cfgrib)
>> sudo apt install libeccodes0

# ---------------------------------------------#
# ---------------------------------------------#
# ---------------------------------------------#



# -----------------------------------------------#
# ------------- WINDOWS & MAC -------------------#
# -----------------------------------------------#

Windows & Mac installation details not yet available. 
The code has been successfully tested on Windows 11.
The code has not been tested on a Mac.


# ---------------------------------------------#
# ---------------------------------------------#
# ---------------------------------------------#


III/ Documentation

SSW is composed of 4 independent modules.
1/ source module calculates the source (initial field in SSW)
2/ terrain generates the relief
3/ propagation is the core of SSW and SSW. It calculates the total electromagnetic field using SSF, SSW or WWP. (WWP-H is the hybridization of SSW and WWP)
SSW is also avalaible in cython (tested on Linux the 2023/04/13). Please see the documentation for compiling.
4/ post-processing plots the result.
5/ GUI contains the graphical user interface. It calls 1/, 2/ and 3/ and plots the results on the GUI directly.

The modules can be called from the GUI (user-friendly mode) or each one independently using the csv files in the inputs directories (expert modes).
If you want to modify the code, the expert mode is recommended.

To call a module, open the corresponding directory with Pycharm and launch the main file.

The detailed documentation of the modules can be generated using teh Doxygen files. One per module.
This documentation is not complete yet. If you need details on a specific module/function, please send an email or ask on the webpage of the GitHub project.

RESET procedure:
If your code does not launch, please reset to the original configuration with the git command:
>> git reset --hard origin/master


IV/ Contact

For questions, suggestions, and so on...
remi.douvenot@enac.fr
or directly on the page of the GitHub project.


V/ How to cite

Rémi Douvenot. SSW-2D: Split-step wavelet in 2D. Software, 2022. ⟨hal-03697711⟩

Hang Zhou, Alexandre Chabory, Rémi Douvenot. A Fast Wavelet-to-Wavelet Propagation Method for the Simulation of Long-Range Propagation in Low Troposphere. 
IEEE Transactions on Antennas and Propagation, 2022, 70, pp.2137-2148. doi: 10.1109/TAP.2021.3118821

Thomas Bonnafont, Rémi Douvenot, Alexandre Chabory, "A local split‐step wavelet method for the long range propagation simulation in 2D",
Radio Science, 2021, 56, doi: 10.1029/2020RS007114

Hang Zhou, Rémi Douvenot, Alexandre Chabory, "Modeling the long-range wave propagation by a split-step wavelet method",
Journal of Computational Physics, 2020, 402, pp.109042. doi: 10.1016/j.jcp.2019.109042

