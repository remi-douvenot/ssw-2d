# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './src_ui/points_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(434, 132)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.lineEditQ = QtWidgets.QLineEdit(Dialog)
        self.lineEditQ.setObjectName("lineEditQ")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEditQ)
        self.lineEditP = QtWidgets.QLineEdit(Dialog)
        self.lineEditP.setObjectName("lineEditP")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEditP)
        self.verticalLayout.addLayout(self.formLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancelPushButton = QtWidgets.QPushButton(Dialog)
        self.cancelPushButton.setObjectName("cancelPushButton")
        self.horizontalLayout.addWidget(self.cancelPushButton)
        self.OKPushButton = QtWidgets.QPushButton(Dialog)
        self.OKPushButton.setObjectName("OKPushButton")
        self.horizontalLayout.addWidget(self.OKPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_3.setText(_translate("Dialog", "Fill in comma-separated latitude and longitude for each point"))
        self.label.setToolTip(_translate("Dialog", "First point"))
        self.label.setText(_translate("Dialog", "P"))
        self.label_2.setToolTip(_translate("Dialog", "Second point"))
        self.label_2.setText(_translate("Dialog", "Q"))
        self.lineEditP.setToolTip(_translate("Dialog", "latitude,longitude"))
        self.cancelPushButton.setText(_translate("Dialog", "Cancel"))
        self.OKPushButton.setText(_translate("Dialog", "OK"))