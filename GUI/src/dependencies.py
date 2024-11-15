# This file is part of SSW-2D.
# SSW-2D is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# SSW-2D is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Foobar. If not, see
# <https://www.gnu.org/licenses/>.

import scipy.constants as cst
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.Qt import QRegularExpression, QRegularExpressionValidator

from src.update_files import update_file
from src.points_dialog import PointsDialog

class Dependencies(object):

    # --- method --- #
    def method_changed(self):
        # --- get method --- #
        method = self.methodComboBox.currentText()
        # write the polarisation value in files
        update_file('method', method, 'propa')

    # --- language --- #
    def language_changed(self):
        # --- get method --- #
        py_or_cy = self.languageComboBox.currentText()
        # write the polarisation value in files
        update_file('py_or_cy', py_or_cy, 'propa')

    # --- frequency --- #
    def frequency_clicked(self):
        # update lambda #
        freq = self.frequencyMHzDoubleSpinBox.value()
        lambda0 = cst.c / (freq * 1e6)  # wavelength in m
        self.lambdaMDoubleSpinBox.setProperty("value", lambda0)
        # source width is defined wrt. lambda0
        self.width_clicked()
        # write the frequency value
        update_file('frequency', freq, 'propa')
        update_file('frequency', freq, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- lambda --- #
    def lambda_clicked(self):
        # update frequency #
        lambda0 = self.lambdaMDoubleSpinBox.value()
        freq = cst.c / lambda0 * 1e-6  # wavelength in m, freq in MHz
        self.frequencyMHzDoubleSpinBox.setProperty("value", freq)
        # write the frequency value
        update_file('frequency', freq, 'propa')
        update_file('frequency', freq, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- polarisation --- #
    def polarisation_changed(self):
        # --- get polarisation --- #
        polarisation = self.polarisationComboBox.currentText()
        # write the polarisation value in files
        update_file('polarisation', polarisation, 'propa')

    # --- N_x --- #
    def n_x_clicked(self):
        # --- update delta_x --- #
        x_max = self.xMaxKmDoubleSpinBox.value()*1e3  # x_max in km
        n_x = self.nXSpinBox.value()
        delta_x = x_max/n_x
        self.deltaXMDoubleSpinBox.setProperty("value", delta_x)
        # @todo Make possible to write several values at once
        # write the x step value
        update_file('x_step', delta_x, 'propa')
        # write the N_x value
        update_file('N_x', n_x, 'propa')
        update_file('N_x', n_x, 'terrain')
        self.relief()

    # --- x_step --- #
    def x_step_clicked(self):
        # --- update N_x --- #
        x_max = self.xMaxKmDoubleSpinBox.value()*1e3  # x_max in km
        delta_x = self.deltaXMDoubleSpinBox.value()
        n_x = int(np.round(x_max / delta_x))

        # write the N_x value
        update_file('N_x', n_x, 'propa')
        # write the x_step value
        update_file('x_step', delta_x, 'propa')
        # write the x_step value in the relief file
        update_file('x_step', delta_x, 'terrain')
        # update N_x on the GUI (and relaunch relief generator)
        self.nXSpinBox.setProperty("value", n_x)
        # terrain button put red
        # self.run_relief.setStyleSheet('QPushButton {background-color: red;}')

    # --- x_max --- #
    def x_max_clicked(self):
        # --- update N_x --- #
        x_max = self.xMaxKmDoubleSpinBox.value()*1e3  # x_max in km
        delta_x = self.deltaXMDoubleSpinBox.value()
        n_x = int(np.round(x_max / delta_x))
        self.nXSpinBox.setProperty("value", n_x)
        # write the N_x value
        update_file('N_x', n_x, 'propa')
        # write the x_max value (in  km)
        # no update for x_max (not written in the config file)
        # update environment plot
        self.plot_environment_in()

    # --- ground type --- #
    def ground_type_changed(self):
        # --- enable/disable epsr and sigma --- #
        groundtype = self.groundTypeComboBox.currentText()
        if groundtype == 'Dielectric':  # disable
            self.sizeImageSpinBox.setEnabled(True)
            self.epsrDoubleSpinBox.setEnabled(True)
            self.sigmaDoubleSpinBox.setEnabled(True)
        elif groundtype == 'PEC':  # enable image only
            self.sizeImageSpinBox.setEnabled(True)
            self.epsrDoubleSpinBox.setEnabled(False)
            self.sigmaDoubleSpinBox.setEnabled(False)
        elif groundtype == 'NoGround':  # enable image only
            self.sizeImageSpinBox.setEnabled(False)
            self.epsrDoubleSpinBox.setEnabled(False)
            self.sigmaDoubleSpinBox.setEnabled(False)
        # write the ground type value in files
        update_file('ground', groundtype, 'propa')

    # --- epsr (relative permittivity) --- #
    def epsr_clicked(self):
        # --- update epsr --- #
        epsr = self.epsrDoubleSpinBox.value()
        # write epsr in the config file
        update_file('epsr', epsr, 'propa')

    # --- sigma (ground conductivity) --- #
    def sigma_clicked(self):
        # --- update sigma --- #
        sigma = self.sigmaDoubleSpinBox.value()
        # write sigma in the config file
        update_file('sigma', sigma, 'propa')

    # --- size image layer --- #
    def image_clicked(self):
        # find wv_L among the input value!!!
        wv_L = self.wvlMaxLevelSpinBox.value()
        # --- update N_im --- #
        n_z = self.nZSpinBox.value()
        image_layer = self.sizeImageSpinBox.value()/100  # image_layer in % of the total size N_z
        n_im = int(np.round(n_z * image_layer))
        remain_im = n_im % 2**wv_L
        if remain_im != 0:
            n_im += 2**wv_L - remain_im
        self.nImageSpinBox.setProperty("value", n_im)
        # write the % of image layer value
        update_file('image size', image_layer, 'propa')

    # --- wavelet max level L --- #
    def wvl_max_level_clicked(self):
        # find wv_L among the input value!!!
        wv_l = self.wvlMaxLevelSpinBox.value()
        # update N_im #
        self.image_clicked()
        # write the wavelet max level L
        update_file('wavelet level', wv_l, 'propa')

    # --- wavelet family --- #
    def wavelet_family_changed(self):
        # --- get family --- #
        wavelet_family = self.wvlFamilyComboBox.currentText()
        # write the wavelet family name in files
        update_file('wavelet family', wavelet_family, 'propa')

    # --- max wavelet compression --- #
    def max_compression_clicked(self):
        # find wv_L among the input value!!!
        max_compression_err = self.maxCompressionDoubleSpinBox.value()
        # write the wavelet max level L
        update_file('Max compression error', max_compression_err, 'propa')

    # --- apodisation type --- #
    def apodisation_changed(self):
        # --- get apodisation window --- #
        wavelet_family = self.apodisationComboBox.currentText()
        # write the apodisation type
        update_file('apodisation window', wavelet_family, 'propa')

    # --- size apodisation (in %) --- #
    def size_apo_clicked(self):
        apo_size = float(self.sizeApoSpinBox.value()) / 100
        # write the apodisation size
        update_file('apodisation size', apo_size, 'propa')

    # --- atmosphere type --- #
    def atm_type_changed(self):
        # --- enable/disable epsr and sigma --- #
        atm_type = self.atmTypeComboBox.currentText()
        if atm_type == 'Homogeneous':  # disable all
            self.c0DoubleSpinBox.setEnabled(False)
            self.deltaDoubleSpinBox.setEnabled(False)
            self.zbDoubleSpinBox.setEnabled(False)
            self.c2DoubleSpinBox.setEnabled(False)
            self.ztDoubleSpinBox.setEnabled(False)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'Linear':  # enable c0 only
            self.c0DoubleSpinBox.setEnabled(True)
            self.deltaDoubleSpinBox.setEnabled(False)
            self.zbDoubleSpinBox.setEnabled(False)
            self.c2DoubleSpinBox.setEnabled(False)
            self.ztDoubleSpinBox.setEnabled(False)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'Evaporation':  # enable c0 and delta
            self.c0DoubleSpinBox.setEnabled(True)
            self.deltaDoubleSpinBox.setEnabled(True)
            self.zbDoubleSpinBox.setEnabled(False)
            self.c2DoubleSpinBox.setEnabled(False)
            self.ztDoubleSpinBox.setEnabled(False)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'Bilinear':  # enable c0, c2 and zt
            self.c0DoubleSpinBox.setEnabled(True)
            self.deltaDoubleSpinBox.setEnabled(False)
            self.zbDoubleSpinBox.setEnabled(False)
            self.c2DoubleSpinBox.setEnabled(True)
            self.ztDoubleSpinBox.setEnabled(True)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'Trilinear':  # enable c0, zb, c2 and zt
            self.c0DoubleSpinBox.setEnabled(True)
            self.deltaDoubleSpinBox.setEnabled(False)
            self.zbDoubleSpinBox.setEnabled(True)
            self.c2DoubleSpinBox.setEnabled(True)
            self.ztDoubleSpinBox.setEnabled(True)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'Double duct':  # enable c0, delta, zb, c2 and zt
            self.c0DoubleSpinBox.setEnabled(True)
            self.deltaDoubleSpinBox.setEnabled(True)
            self.zbDoubleSpinBox.setEnabled(True)
            self.c2DoubleSpinBox.setEnabled(True)
            self.ztDoubleSpinBox.setEnabled(True)
            self.atmosphereDateTimeEdit.setEnabled(False)
        elif atm_type == 'ERA5':  # enable file name only
            self.c0DoubleSpinBox.setEnabled(False)
            self.deltaDoubleSpinBox.setEnabled(False)
            self.zbDoubleSpinBox.setEnabled(False)
            self.c2DoubleSpinBox.setEnabled(False)
            self.ztDoubleSpinBox.setEnabled(False)
            self.atmosphereDateTimeEdit.setEnabled(True)
        # write the ground type value in files
        update_file('atmosphere', atm_type, 'propa')
        self.plot_environment_in()

    # --- c0 (atm) --- #
    def c0_clicked(self):
        # --- update c0 --- #
        c0 = self.c0DoubleSpinBox.value()  # c0 in M-unit/m
        # write the c0 value
        update_file('c0', c0, 'propa')
        self.plot_environment_in()

    # --- delta (atm) --- #
    def delta_clicked(self):
        # --- update delta --- #
        delta = self.deltaDoubleSpinBox.value()  # delta in m
        # write the delta value
        update_file('delta', delta, 'propa')
        self.plot_environment_in()

    # --- zb (atm) --- #
    def zb_clicked(self):
        # --- update delta --- #
        zb = self.zbDoubleSpinBox.value()  # zb in m
        # write the delta value
        update_file('zb', zb, 'propa')
        self.plot_environment_in()

    # --- c2 (atm) --- #
    def c2_clicked(self):
        # --- update c2 --- #
        c2 = self.c2DoubleSpinBox.value()  # c2 in M-unit/m
        # write the c0 value
        update_file('c2', c2, 'propa')
        self.plot_environment_in()

    # --- zt (atm) --- #
    def zt_clicked(self):
        # --- update delta --- #
        zt = self.ztDoubleSpinBox.value()  # zb in m
        # write the delta value
        update_file('zt', zt, 'propa')
        self.plot_environment_in()

    # --- turbulence --- #
    def turbulence_yes_no(self):
        turbu_type = self.turbuComboBox.currentText()
        if turbu_type == 'N':  # disable all
            self.Cn2DoubleSpinBox.setEnabled(False)
            self.L0DoubleSpinBox.setEnabled(False)
        else:
            self.Cn2DoubleSpinBox.setEnabled(True)
            self.L0DoubleSpinBox.setEnabled(True)
        update_file('turbulence', turbu_type, 'propa')
        self.plot_environment_in()

    # --- Cn2 (exponent) --- #
    def Cn2_clicked(self):
        # --- update Cn2 --- #
        Cn2 = self.Cn2DoubleSpinBox.value()  # Cn2 exponent
        # write the Cn2
        update_file('Cn2', Cn2, 'propa')
        self.plot_environment_in()

    # --- L0 --- #
    def L0_clicked(self):
        # --- update L0 --- #
        L0 = self.L0DoubleSpinBox.value()  # L0 in m
        # write the L0
        update_file('L0', L0, 'propa')
        self.plot_environment_in()

    # --- N_z --- #
    def n_z_clicked(self):
        # --- update z_step --- #
        z_max = self.zMaxMDoubleSpinBox.value()
        n_z = self.nZSpinBox.value()
        delta_z = z_max / n_z
        self.deltaZMDoubleSpinBox.setProperty("value", delta_z)
        # write the z_step value in files
        update_file('z_step', delta_z, 'propa')
        update_file('z_step', delta_z, 'source')
        # write the n_z value in files
        update_file('N_z', n_z, 'propa')
        update_file('N_z', n_z, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')
        self.run_simulation.setStyleSheet('QPushButton {background-color: red;}')

    # --- z_step --- #
    def z_step_clicked(self):
        # --- update N_z --- #
        z_max = self.zMaxMDoubleSpinBox.value()
        delta_z = self.deltaZMDoubleSpinBox.value()
        n_z = int(np.round(z_max / delta_z))
        self.nZSpinBox.setProperty("value", n_z)
        # write the z_step value in files
        update_file('z_step', delta_z, 'propa')
        update_file('z_step', delta_z, 'source')
        # write the n_z value in files
        update_file('N_z', n_z, 'propa')
        update_file('N_z', n_z, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- z_max --- #
    def z_max_clicked(self):

        # --- update N_z --- #
        z_max = self.zMaxMDoubleSpinBox.value()
        delta_z = self.deltaZMDoubleSpinBox.value()
        n_z = int(np.round(z_max / delta_z))
        self.nZSpinBox.setProperty("value", n_z)
        # write the n_z value in files
        update_file('N_z', n_z, 'propa')
        update_file('N_z', n_z, 'source')
        self.plot_environment_in()
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- x_s --- #
    def x_s_clicked(self):
        # --- update x_s --- #
        x_s = self.x_sDoubleSpinBox.value()
        # write the x_s value in files
        update_file('x_s', -x_s, 'source')  # Warning: x_s is negative in the input file
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- z_s --- #
    def z_s_clicked(self):
        # --- update z_s --- #
        z_s = self.z_sDoubleSpinBox.value()
        # write the z_s value in files
        update_file('z_s', z_s, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- width --- #
    def width_clicked(self):
        # --- update width --- #
        width = self.widthDoubleSpinBox.value()
        lambda0 = self.lambdaMDoubleSpinBox.value()
        width_in_m = width * lambda0
        # write the z_s value in files
        update_file('W0', width_in_m, 'source')
        self.run_source.setStyleSheet('QPushButton {background-color: red;}')

    # --- dynamic --- #
    def dynamic_clicked(self):
        # --- update dynamic --- #
        dynamic = self.dynamicSpinBox.value()
        # write the dynamic value in file
        update_file('dynamic', dynamic, 'propa')
        # update the final plot
        v_max = self.ax_final.get_xlim()[1]
        v_min = v_max - dynamic
        self.ax_final.set_xlim((v_min, v_max))
        self.canvas_final.draw()
        # update the source plot
        v_max = self.ax_source.get_xlim()[1]
        v_min = v_max - dynamic
        self.ax_source.set_xlim((v_min, v_max))
        self.canvas_source.draw()
        # update the field plot
        self.plot_field_in()

    # --- relief type --- #
    def relief_type_changed(self):
        # --- enable/disable relief generation --- #
        relief_type = self.reliefTypeComboBox.currentText()
        if relief_type == 'Plane':  # no relief, disable
            self.maxReliefDoubleSpinBox.setEnabled(False)
            self.nIterationsSpinBox.setEnabled(False)
            self.centerReliefDoubleSpinBox.setEnabled(False)
            self.widthReliefDoubleSpinBox.setEnabled(False)
        elif relief_type == 'Superposed':  # enable max relief and iterations
            self.maxReliefDoubleSpinBox.setEnabled(True)
            self.nIterationsSpinBox.setEnabled(True)
            self.centerReliefDoubleSpinBox.setEnabled(False)
            self.widthReliefDoubleSpinBox.setEnabled(False)
        elif relief_type == 'Triangle':  # enable max relief and iterations
            self.maxReliefDoubleSpinBox.setEnabled(True)
            self.nIterationsSpinBox.setEnabled(False)
            self.centerReliefDoubleSpinBox.setEnabled(True)
            self.widthReliefDoubleSpinBox.setEnabled(True)

        # write the ground type value in files
        update_file('type', relief_type, 'terrain')
        # put relief button green
        self.run_relief.setStyleSheet('QPushButton {background-color: green;}')

    # --- max relief altitude --- #
    def max_relief_clicked(self):
        # --- update width --- #
        max_relief = self.maxReliefDoubleSpinBox.value()
        # write the max_relief value in terrain file
        update_file('z_max_relief', max_relief, 'terrain')
        # put relief button green
        self.run_relief.setStyleSheet('QPushButton {background-color: green;}')

    # --- max relief altitude --- #
    def relief_iterations_clicked(self):
        # --- update max relief --- #
        relief_iterations = self.nIterationsSpinBox.value()
        # write the max_relief value in terrain file
        update_file('iterations', relief_iterations, 'terrain')
        # put relief button green
        self.run_relief.setStyleSheet('QPushButton {background-color: green;}')

    # --- center relief --- #
    def relief_center_clicked(self):
        # --- read Delta_x --- #
        x_step = self.deltaXMDoubleSpinBox.value()
        # @TODO Center and width defined in meters in the csv file
        # --- update relief center expressed in number of points (not in m) --- #
        relief_center = float(self.centerReliefDoubleSpinBox.value())
        # write the center relief value in terrain file
        update_file('center', relief_center, 'terrain')

    # --- relief width --- #
    def relief_width_clicked(self):
        # --- read Delta_x --- #
        x_step = self.deltaXMDoubleSpinBox.value()
        # --- update relief width expressed in number of points (not in m) --- #
        relief_width = float(self.widthReliefDoubleSpinBox.value())
        # write the max_relief value in terrain file
        update_file('width', relief_width, 'terrain')
        update_file('x_step', x_step, 'terrain')

    # --- output type --- #
    def output_changed(self):
        # --- get apodisation window --- #
        output_type = self.outputComboBox.currentText()
        # --- refresh plots --- #
        self.plot_field_in()

    def open_points_dialog(self):
        dialog = PointsDialog(self)
        dialog.exec() # dialog sets it's parent P and Q members
        # Save P and Q into the configuration files
        to_str = lambda l: ';'.join([str(e) for e in l])
        update_file('P', to_str(self.P), 'propa')
        update_file('Q', to_str(self.Q), 'propa')
        update_file('P', to_str(self.P), 'terrain')
        update_file('Q', to_str(self.Q), 'terrain')

    def datetime_changed(self, datetime:QDateTime):
        # Prevent user from setting minutes and seconds
        currentTime = datetime.time() # get time
        currentTime.setHMS(currentTime.hour(), 0, 0) # set minutes and hours to 0
        datetime.setTime(currentTime) # set the upsated time
        self.atmosphereDateTimeEdit.setDateTime(datetime) # set datetime in the UI
        update_file('atmosphere_datetime', datetime.toString(Qt.ISODate)[:-1], 'propa') # Remove the Z at the end of the string ([:-1])

    # --- Initialise all the values --- #
    def initialise(self):
        file_propa = '../propagation/inputs/configuration.csv'
        file_source = '../source/inputs/configuration.csv'
        file_relief = '../terrain/inputs/conf_terrain.csv'

        # --- Initialise main config --- #
        with open(file_propa) as file_to_read:
            # reading the csv file as a dataframe
            dataframe = pd.read_csv(file_to_read)
            # create a Series with it
            serie = pd.Series(data=dataframe.iloc[:, 1].values, index=dataframe.iloc[:, 0].values)
            # method
            self.methodComboBox.setCurrentText(serie.loc['method'])
            # language
            self.languageComboBox.setCurrentText(serie.loc['py_or_cy'])
            # N_x
            self.nXSpinBox.setProperty("value", serie.loc['N_x'])
            # N_z
            self.nZSpinBox.setProperty("value", serie.loc['N_z'])
            # x_step
            self.deltaXMDoubleSpinBox.setProperty("value", serie.loc['x_step'])
            # z_step
            self.deltaZMDoubleSpinBox.setProperty("value", serie.loc['z_step'])
            # x_max
            x_max = float(serie.loc['x_step']) * float(serie.loc['N_x']) * 1e-3  # in km
            self.xMaxKmDoubleSpinBox.setProperty("value", str(x_max))
            # z_max
            z_max = float(serie.loc['z_step']) * int(serie.loc['N_z'])
            self.zMaxMDoubleSpinBox.setProperty("value", z_max)
            # frequency
            self.frequencyMHzDoubleSpinBox.setProperty("value", serie.loc['frequency'])
            # lambda
            lambda0 = cst.c/float(serie.loc['frequency'])*1e-6  # freq in MHz
            self.lambdaMDoubleSpinBox.setProperty("value", lambda0)
            # ground
            self.polarisationComboBox.setCurrentText(serie.loc['polarisation'])
            # Max compression error
            self.maxCompressionDoubleSpinBox.setProperty("value", serie.loc['Max compression error'])
            # wavelet level
            self.wvlMaxLevelSpinBox.setProperty("value", serie.loc['wavelet level'])
            # wavelet family
            self.wvlFamilyComboBox.setCurrentText(serie.loc['wavelet family'])
            # apodisation window
            self.apodisationComboBox.setProperty("value", serie.loc['apodisation window'])
            # apodisation % in z
            self.sizeApoSpinBox.setProperty("value",  float(serie.loc['apodisation size'])*100)  # from value to %
            # image layer % in z
            self.sizeImageSpinBox.setProperty("value", float(serie.loc['image size'])*100)  # from value to %
            self.image_clicked()  # initialise nb points in image layer
            # ground
            self.groundTypeComboBox.setCurrentText(serie.loc['ground'])
            self.ground_type_changed()
            # epsr
            self.epsrDoubleSpinBox.setProperty("value", serie.loc['epsr'])
            # sigma
            self.sigmaDoubleSpinBox.setProperty("value", serie.loc['sigma'])
            # atmosphere type
            self.atmTypeComboBox.setCurrentText(serie.loc['atmosphere'])
            # c0
            self.c0DoubleSpinBox.setProperty("value", serie.loc['c0'])
            # delta
            self.deltaDoubleSpinBox.setProperty("value", serie.loc['delta'])
            # zb
            self.zbDoubleSpinBox.setProperty("value", serie.loc['zb'])
            # c2
            self.c2DoubleSpinBox.setProperty("value", serie.loc['c2'])
            # zt
            self.ztDoubleSpinBox.setProperty("value", serie.loc['zt'])
            # dynamic
            self.dynamicSpinBox.setProperty("value", serie.loc['dynamic'])
            # turbulence
            self.turbuComboBox.setCurrentText(serie.loc['turbulence'])
            #Cn2
            self.Cn2DoubleSpinBox.setProperty("value", serie.loc['Cn2'])
            #L0
            self.L0DoubleSpinBox.setProperty("value", serie.loc['L0'])
            # Store P and Q as attributes - duplicated code
            self.P = tuple([float(l) for l in serie.loc['P'].split(';')])
            self.Q = tuple([float(l) for l in serie.loc['Q'].split(';')])
            # Load date into the UI
            datetime = QDateTime.fromString(serie.loc['atmosphere_datetime'], Qt.ISODate)
            self.atmosphereDateTimeEdit.setDateTime(datetime)

        # --- Initialise source --- #
        with open(file_source) as file_to_read:
            # reading the csv file as a dataframe
            dataframe = pd.read_csv(file_to_read)
            # create a Series with it
            serie = pd.Series(data=dataframe.iloc[:, 1].values, index=dataframe.iloc[:, 0].values)
            # type
            self.sourceTypeComboBox.setCurrentText(serie.loc['type'])
            # x_s
            x_s = serie.loc['x_s']  # then remove the "-"
            self.x_sDoubleSpinBox.setProperty("value", x_s[1:])
            # z_s
            self.z_sDoubleSpinBox.setProperty("value", float(serie.loc['z_s']))
            # W0 (width of the belt for CSP source)
            w0_lambda = float(serie.loc['W0']) / lambda0
            # print('W0_lambda', w0_lambda)
            self.widthDoubleSpinBox.setProperty("value", w0_lambda)
            # @todo P_Tx
            # self.nZSpinBox.setProperty("value", serie.loc['N_z'])
            # @todo G_Tx
            # self.nZSpinBox.setProperty("value", serie.loc['N_z'])

        # --- Initialise relief --- #
        with open(file_relief) as file_to_read:
            # reading the csv file as a dataframe
            dataframe = pd.read_csv(file_to_read)
            # create a Series with it
            serie = pd.Series(data=dataframe.iloc[:, 1].values, index=dataframe.iloc[:, 0].values)
            # print(serie)
            # type
            self.reliefTypeComboBox.setCurrentText(serie.loc['type'])
            # max relief
            self.maxReliefDoubleSpinBox.setProperty("value", float(serie.loc['z_max_relief']))
            # number of iterations
            self.nIterationsSpinBox.setProperty("value", int(serie.loc['iterations']))
            # width of the relief
            self.widthReliefDoubleSpinBox.setProperty("value", float(serie.loc['width']))
            # center of the relief
            self.centerReliefDoubleSpinBox.setProperty("value", float(serie.loc['center']))
