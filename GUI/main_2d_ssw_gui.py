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
# @mainpage        GUI: Graphic User Interface for SSW-2D
# @author          Rémi Douvenot, ENAC
# @date            17/06/22
# @version
#
# @section intro   introduction
#                  This document describes the code for the GUI SSW-2D code.
#
# @section prereq   Prerequisites.
#                   Python packages: numpy, scipy, pywavelets, matplolib, PyQT5, Pandas
#
# @section install Installation procedure
#                  No installation. Just run the main program.
#
# @section run      Run GUI for SSW-2D
#                   SSW-2D can be run through this GUI. Just run the main, fill the desired options, and press \n
#                   - "run source" to generate the initial field \n
#                   - "run relief" to generate the relief \n
#                   - "run simulation" to launch the propagation \n
#                   - (the atmosphere is automatically generated with the options are chosen.)
#
##

##
# @file main_2d_ssw_gui.py
#
# @package: main_2d_ssw_hmi
# @author: R. Douvenot
# @date: 20/07/2021
# @version: work in progress
#
# @brief GUI for 2D SSW
# @description GUI for 2D SSW. Generate the main class from ui (created with QT designer) using \n
# >> pyuic5 -o ../src/gui_ssw_2d.py gui_ssw_2d.ui
#
# Any new parameter of the GUI must appear 3 times in this code.
# 1. In this file: --- declare dependencies --- to assign an action to the button
# 2. In dependencies.py: define the function that makes the actions. At least one: write in the configuration file
# 3. In dependencies.py: in initialise function, such that the value in read when the GUI is opened
##

# !/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
import pandas as pd
import sys
import os
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QGraphicsScene, QGraphicsView, \
    QGridLayout, QVBoxLayout, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.Qt import QProcess, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Local Module Imports
import src.plots
from src.gui_ssw_2d import Ui_MainWindow
from src.dependencies import Dependencies
from src.plots import Plots
import csv

class Window(QMainWindow, Ui_MainWindow, Dependencies, Plots):

    def __init__(self, parent=None):
        # Since Window in derived from 4 different classes, we need to call each individual constructor
        # Plots class does not have a ctor/init function
        # Note : constructor/initialization calls have 'static' syntax here, but they do the same thing
        QMainWindow.__init__(self, parent)
        Ui_MainWindow.setupUi(self, self)
        Dependencies.initialise(self)
        self.environment_colorbar = None

        # @TODO: make expandable and manipulable figures (cf. example Alex)

        # -------------------------- #
        # -------- Processes ------- #
        # -------------------------- #
        self.sim_process = QProcess(self)
        self.terrain_process = QProcess(self)
        self.source_process = QProcess(self)

        # -------------------------- #
        # --- Link GUI and plots --- #
        # -------------------------- #

        # --- Link total field plot to GUI --- #
        self.fieldGridLayout = QGridLayout(self.field_window)
        self.fieldGridLayout.setContentsMargins(0, 0, 0, 0)
        self.fieldGridLayout.setObjectName("fieldGridLayout")
        self.scene = QGraphicsScene(self)
        self.field_window.setScene(self.scene)
        # self.fieldGridLayout.addWidget(self.field_window, 0, 0, 1, 1)
        # @todo Pass the plots in "constrained_layout" as soon as it works with removing colorbar
        self.figure_field = plt.figure(figsize=(8.2, 3.8), tight_layout=True)
        self.canvas_field = FigureCanvas(self.figure_field)
        self.scene.addWidget(self.canvas_field)
        self.ax_field = self.figure_field.add_subplot(1, 1, 1)

        # --- Link final field plot to GUI --- #
        self.final_fieldGridLayout = QGridLayout(self.final_field_window)
        self.final_fieldGridLayout.setContentsMargins(0, 0, 0, 0)
        self.final_fieldGridLayout.setObjectName("fieldGridLayout")
        self.scene = QGraphicsScene(self)
        self.final_field_window.setScene(self.scene)
        # self.final_fieldGridLayout.addWidget(self.final_field_window, 0, 0, 1, 1)
        self.figure_final = plt.figure(figsize=(2.7, 3.8), tight_layout=True)
        self.canvas_final = FigureCanvas(self.figure_final)
        self.scene.addWidget(self.canvas_final)
        self.ax_final = self.figure_final.add_subplot(1, 1, 1)

        # --- Link source plot to GUI --- #
        self.sourceGridLayout = QGridLayout(self.source_window)
        self.sourceGridLayout.setContentsMargins(0, 0, 0, 0)
        self.sourceGridLayout.setObjectName("sourceGridLayout")
        self.scene = QGraphicsScene(self)
        self.source_window.setScene(self.scene)
        # self.sourceGridLayout.addWidget(self.source_window, 0, 0, 1, 1)
        self.figure_source = plt.figure(figsize=(1.88, 3.8), tight_layout=True)
        self.canvas_source = FigureCanvas(self.figure_source)
        self.scene.addWidget(self.canvas_source)
        self.ax_source = self.figure_source.add_subplot(1, 1, 1)

        # --- Link environment plot to GUI --- #
        self.environmentGridLayout = QGridLayout(self.environment_window)
        self.environmentGridLayout.setContentsMargins(0, 0, 0, 0)
        self.environmentGridLayout.setObjectName("environmentGridLayout")
        self.scene = QGraphicsScene(self)
        self.environment_window.setScene(self.scene)
        # self.environmentGridLayout.addWidget(self.environment_window, 0, 0, 1, 1)
        # @todo Pass the plots in "constrained_layout" as soon as it works with removing colorbar
        self.figure_environment = plt.figure(figsize=(8.2, 2.5), tight_layout=True)
        self.canvas_environment = FigureCanvas(self.figure_environment)
        self.scene.addWidget(self.canvas_environment)
        self.ax_environment = self.figure_environment.add_subplot(1, 1, 1) # 1 row, 1 column, 1st figure
        # set the enabled / disabled buttons in the atmosphere tab
        self.atm_type_changed()

        # --- LOGOS --- #
        # GNU GPL
        pix_gpl = QPixmap('./src/GPL_logo_small2.png')
        # pix.scaled()
        item = QGraphicsPixmapItem(pix_gpl)
        scene = QGraphicsScene(self)
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        # ENAC
        pix_enac = QPixmap('./src/ENAC_logo_small.png')
        # pix.scaled()
        item = QGraphicsPixmapItem(pix_enac)
        scene = QGraphicsScene(self)
        scene.addItem(item)
        self.graphicsLogoEnac.setScene(scene)
        # Republique francaise
        pix_rf = QPixmap('./src/RF_logo_small.png')
        # pix.scaled()
        item = QGraphicsPixmapItem(pix_rf)
        scene = QGraphicsScene(self)
        scene.addItem(item)
        self.graphicsLogoMinister.setScene(scene)
        # SSW-2D
        pix_ssw = QPixmap('./src/logo_SSW_2D_small.png')
        # pix.scaled()
        item = QGraphicsPixmapItem(pix_ssw)
        scene = QGraphicsScene(self)
        scene.addItem(item)
        self.graphicsSSW.setScene(scene)
        # ------------- #

        # ---------- END ----------- #
        # --- Link GUI and plots --- #
        # -------------------------- #

        # ---------------------- #
        # --- Action buttons --- #
        # ---------------------- #

        self.run_simulation.clicked.connect(self.ssw)
        self.run_source.clicked.connect(self.source)
        self.run_relief.clicked.connect(self.relief)
        self.pushButtonSource.clicked.connect(self.plot_source_out)
        self.pushButtonField.clicked.connect(self.plot_field_out)
        self.pushButtonFinal.clicked.connect(self.plot_final_out)
        self.pushButtonEnvironment.clicked.connect(self.plot_environment_out)
        self.pushButtonRefractivity.clicked.connect(self.plot_refractivity_out)
        self.setPointsButton.clicked.connect(self.open_points_dialog)

        # -------- END --------- #
        # --- Action buttons --- #
        # ---------------------- #

        # ---------------------------- #
        # --- Declare dependencies --- #
        # ---------------------------- #

        # --- Main features --- #
        # Method
        self.methodComboBox.currentTextChanged.connect(self.method_changed)
        # Language
        self.languageComboBox.currentTextChanged.connect(self.language_changed)
        # Frequency and wavelength
        self.frequencyMHzDoubleSpinBox.valueChanged.connect(self.frequency_clicked)
        self.lambdaMDoubleSpinBox.valueChanged.connect(self.lambda_clicked)
        self.polarisationComboBox.currentTextChanged.connect(self.polarisation_changed)
        # horizontal steps: number and size
        self.nXSpinBox.valueChanged.connect(self.n_x_clicked)
        self.deltaXMDoubleSpinBox.valueChanged.connect(self.x_step_clicked)
        self.xMaxKmDoubleSpinBox.valueChanged.connect(self.x_max_clicked)
        # vertical steps: number and size
        self.nZSpinBox.valueChanged.connect(self.n_z_clicked)
        self.deltaZMDoubleSpinBox.valueChanged.connect(self.z_step_clicked)
        self.zMaxMDoubleSpinBox.valueChanged.connect(self.z_max_clicked)

        # --- Source --- #
        self.z_sDoubleSpinBox.valueChanged.connect(self.z_s_clicked)
        self.x_sDoubleSpinBox.valueChanged.connect(self.x_s_clicked)
        self.widthDoubleSpinBox.valueChanged.connect(self.width_clicked)

        # --- Ground --- #
        self.groundTypeComboBox.currentTextChanged.connect(self.ground_type_changed)
        self.sizeImageSpinBox.valueChanged.connect(self.image_clicked)
        self.sigmaDoubleSpinBox.valueChanged.connect(self.sigma_clicked)
        self.epsrDoubleSpinBox.valueChanged.connect(self.epsr_clicked)

        # --- Wavelets --- #
        self.wvlMaxLevelSpinBox.valueChanged.connect(self.wvl_max_level_clicked)
        self.wvlFamilyComboBox.currentTextChanged.connect(self.wavelet_family_changed)
        self.maxCompressionDoubleSpinBox.valueChanged.connect(self.max_compression_clicked)

        # --- Apodisation --- #
        self.apodisationComboBox.currentTextChanged.connect(self.apodisation_changed)
        self.sizeApoSpinBox.valueChanged.connect(self.size_apo_clicked)

        # --- wvl propagator --- #
        self.wvlPropagatorComboBox.currentTextChanged.connect(self.wvl_propagator_changed)

        # --- Atmosphere --- #
        self.atmTypeComboBox.currentTextChanged.connect(self.atm_type_changed)
        self.c0DoubleSpinBox.valueChanged.connect(self.c0_clicked)
        self.deltaDoubleSpinBox.valueChanged.connect(self.delta_clicked)
        self.zbDoubleSpinBox.valueChanged.connect(self.zb_clicked)
        self.c2DoubleSpinBox.valueChanged.connect(self.c2_clicked)
        self.ztDoubleSpinBox.valueChanged.connect(self.zt_clicked)
        self.atmosphereDateTimeEdit.dateTimeChanged.connect(self.datetime_changed)

        # --- turbulence --- #
        self.turbuComboBox.currentTextChanged.connect(self.turbulence_yes_no)
        self.Cn2DoubleSpinBox.valueChanged.connect(self.Cn2_clicked)
        self.L0DoubleSpinBox.valueChanged.connect(self.L0_clicked)

        # --- Relief --- #
        self.reliefTypeComboBox.currentTextChanged.connect(self.relief_type_changed)
        self.maxReliefDoubleSpinBox.valueChanged.connect(self.max_relief_clicked)
        self.nIterationsSpinBox.valueChanged.connect(self.relief_iterations_clicked)
        self.widthReliefDoubleSpinBox.valueChanged.connect(self.relief_width_clicked)
        self.centerReliefDoubleSpinBox.valueChanged.connect(self.relief_center_clicked)
        # set the enabled / disabled buttons in the atmosphere tab
        self.relief_type_changed()

        # --- Outputs --- #
        # @todo: from E to F, L, S
        self.dynamicSpinBox.valueChanged.connect(self.dynamic_clicked)
        self.outputComboBox.currentTextChanged.connect(self.output_changed)
        # ----------- END ------------ #
        # --- Declare dependencies --- #
        # ---------------------------- #

    def ssw(self):
        # Get the resolution method from the UI
        method = self.methodComboBox.currentText()
        start_message = "Field calculation using "+method+" -- in progress"
        end_message = "Field calculation using "+method+" -- Successfully finished"
        error_message = "Field calculation using "+method+" -- Error. Do not consider the display"
        groundType = self.groundTypeComboBox.currentText()
        if method == 'WWP':
            if groundType == 'Dielectric':
                error_message = "Error with WWP, Dielectric ground not yet available in WWP"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('Error with WWP, ground not accounted. Do not consider the display')
            elif groundType == 'PEC':
                error_message = "Error with WWP, PEC ground not yet available in WWP"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('Error with WWP, ground not accounted. Do not consider the display')
        elif method == 'SSF':
            if groundType == 'Dielectric':
                error_message = "Dielectric ground not yet available in SSF"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('Dielectric ground not yet available in SSF')
        elif method == 'WWP-H':
            if groundType == 'No Ground':
                error_message = "There is no ground, Please use WWP instead of WWP-H"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('There is no ground, Please use WWP instead of WWP-H')
            elif groundType == 'Dielectric':
                error_message = "Dielectric ground not yet available in WWP-H"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('Dielectric ground not yet available in WWP-H')
        elif method == 'WGM':
            if groundType != 'No Ground':
                error_message = "WGM only available for No Ground"
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('WGM only available for No Ground')
        freq = self.frequencyMHzDoubleSpinBox.value()
        apod = self.sizeApoSpinBox.value()
        file_source_output_config = '../source/outputs/configuration.csv'
        f_source_config = open(file_source_output_config, newline='')
        file_tmp = csv.reader(f_source_config)
        for row in file_tmp:
            if row[0] == 'frequency':
                freq1 = np.float64(row[1])
                if freq != freq1:
                    error_message = "The value of the Frequency does NOT match with source generation. Please Click on the Run Source Button Before Running the Simulation."
                    self.informationTextBrowser.setPlainText(error_message)
                    raise ValueError(['frequency ', freq, ' MHz value does not match with source generation', freq1, ' MHz'])
            elif row[0] == 'z_step':
                z_step = np.float64(row[1])
            elif row[0] == 'N_z':
                N_z = int(row[1])
            elif row[0] == 'z_s':
                z_s = np.float64(row[1])
                file_relief_config = '../terrain/outputs/z_relief.csv'
                relief_config = open(file_relief_config, newline='')
                file = csv.reader(relief_config)
                for l in file:
                    relief0 = np.float64(l[0])
                    if (z_s + relief0) >= N_z * z_step * (1 - apod * 1e-2) or (groundType == 'No Ground' and z_s <= 400 * apod * 1e-2):
                        error_message = "Warning: The Source is in the Apodization Zone"
                        self.informationTextBrowser.setPlainText(error_message)
                        raise ValueError('The Source is in the Apodization Zone')
        reliefType = self.reliefTypeComboBox.currentText()
        if reliefType == 'Superposed':
            Iteration = self.nIterationsSpinBox.value()
            if Iteration == 0:
                error_message = "Warning : Iterations(scales) = 0. Please change the value."
                self.informationTextBrowser.setPlainText(error_message)
                raise ValueError('zero-size array to reduction operation minimum which has no identity')
        self.run_simulation.setStyleSheet('QPushButton {background-color: red;}')
        self.informationTextBrowser.setPlainText(start_message)
        self.informationTextBrowser.repaint()
        # check for modulo
        n_z = self.nZSpinBox.value()
        wv_l = self.wvlMaxLevelSpinBox.value()
        flag_error = check_modulo(n_z, wv_l)
        turbu_type = self.turbuComboBox.currentText()
        # ground_type = self.groundTypeComboBox.currentText()
        #flag_error_turbu = check_turbu(turbu_type,ground_type)
        if flag_error:
            self.informationTextBrowser.setPlainText("ERROR: N_z must be multiple of 2^L (max wavelet level)")
            raise (ValueError(['N_z must be multiple of 2^L (max wavelet level)']))
        #if flag_error_turbu:
            #self.informationTextBrowser.setPlainText("ERROR: Ground type must be No Ground if turbulence is True")
            #raise (ValueError(['Ground type must be No Ground if turbulence is True']))
        # main program: SSW propagation

        # Check if a process is already running, kill it otherwise
        if self.sim_process.state() == QProcess.Running:
            self.sim_process.kill()

        # Create a new process
        self.sim_process = QProcess(self)
        # When the process has exited, run self.simulation_finished()
        self.sim_process.finished.connect(self.simulation_finished)
        # When process stdout has messages or errors, print them in the console
        self.sim_process.readyReadStandardOutput.connect(lambda: self.log_stdout(self.sim_process))
        self.sim_process.readyReadStandardError.connect(lambda: self.log_stderr(self.sim_process))
        # Set our process executable, sys.executable gives a /path/to/python or C:\path\to\python.exe
        self.sim_process.setProgram(sys.executable) # platform-independent
        # Set the directory in which the command will be launched
        self.sim_process.setWorkingDirectory('../propagation')
        # Give 1 argument to the programm (script name), has to be a List[str]
        self.sim_process.setArguments(['main_propagation.py'])
        # Start the process
        self.sim_process.start()

    def log_stdout(self, process):
        # QProcess.readAllStandardOutput() returns a QByteArray
        # QByteArray.data() returns python bytes
        # bytes.decode() returns a str
        self.informationTextBrowser.append(process.readAllStandardOutput().data().decode())

    def log_stderr(self, process):
        # QProcess.readAllStandardError() returns a QByteArray
        # QByteArray.data() returns python bytes
        # bytes.decode() returns a str
        self.informationTextBrowser.append(process.readAllStandardError().data().decode())

    def simulation_finished(self, exitcode, exit_status):
        # plot on the GUI
        self.plot_field_in()
        # set button color
        self.run_simulation.setStyleSheet('QPushButton {background-color: green;}')

    def source(self):
        # start message
        self.informationTextBrowser.setPlainText("Source calculation -- in progress")
        self.informationTextBrowser.repaint()

        # See self.ssw() for explanations on the code below
        if self.source_process.state() == QProcess.Running:
            self.source_process.kill()

        self.source_process = QProcess(self)
        self.source_process.finished.connect(self.source_finished)
        self.source_process.readyReadStandardOutput.connect(lambda: self.log_stdout(self.source_process))
        self.source_process.readyReadStandardError.connect(lambda: self.log_stderr(self.source_process))
        self.source_process.setProgram(sys.executable) # sys.executable gives a path to the python interpreter (platform-independant)
        self.source_process.setWorkingDirectory('../source')
        self.source_process.setArguments(['main_source.py'])
        self.source_process.start()

    def source_finished(self, exitcode, exit_status):
        # update plot
        self.plot_source_in()

        # end message
        self.informationTextBrowser.setPlainText("Source calculation -- finished")
        self.run_source.setStyleSheet('QPushButton {background-color: lightgray;}')
        self.run_simulation.setStyleSheet('QPushButton {background-color: green;}')

    def relief(self):
        # See self.ssw() for explanations on the code below
        if self.terrain_process.state() == QProcess.Running:
            self.terrain_process.kill()

        self.terrain_process = QProcess(self)
        self.terrain_process.finished.connect(self.terrain_finished)
        self.terrain_process.readyReadStandardOutput.connect(lambda: self.log_stdout(self.terrain_process))
        self.terrain_process.readyReadStandardError.connect(lambda: self.log_stderr(self.terrain_process))
        self.terrain_process.setProgram(sys.executable) # sys.executable gives a path to the python interpreter (platform-independant)
        self.terrain_process.setWorkingDirectory('../terrain')
        self.terrain_process.setArguments(['main_terrain.py'])
        self.terrain_process.start()

    def terrain_finished(self, exitcode, exit_status):
        # plot on the GUI
        self.plot_environment_in()
        # and update source plot
        self.plot_source_in()

        # end message
        self.informationTextBrowser.append("Terrain generation -- finished")
        self.run_relief.setStyleSheet('QPushButton {background-color: lightgray;}')

    def about(self):
        QMessageBox.about(
            self,
            "About Sample Editor",
            "<p>A sample text editor app built with:</p>"
            "<p>A sample text editor app built with:</p>"
            "<p>- PyQt</p>"
            "<p>- Qt Designer</p>"
            "<p>- Python</p>",
        )


# Function to check that n_z number is a modulo of 2^L (necessary condition for propagation)
def check_modulo(n_z, wv_l):
    # --- Check the size of the vectors, multiple of 2^n --- #
    n_scaling_fct = 2 ** wv_l
    modulo_nz = n_z % n_scaling_fct
    if modulo_nz != 0:
        flag_error = True
    else:
        flag_error = False
    return flag_error

"""
def check_turbu(turbu_type, ground_type):
    # --- Check the size of the vectors, multiple of 2^n --- #
    if turbu_type == 'Y' and ground_type != 'No Ground':
        flag_error = True
    else:
        flag_error = False
    return flag_error
"""

if __name__ == "__main__":

    app = QApplication(sys.argv)
    # Create GUI application
    win = Window()
    win.show()
    sys.exit(app.exec())
# Create GUI application


