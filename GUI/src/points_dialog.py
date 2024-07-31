from PyQt5.QtWidgets import QDialog, QErrorMessage
from PyQt5.Qt import QRegularExpressionValidator, QRegularExpression
from geographiclib.geodesic import Geodesic

from src.gui_dialog import Ui_Dialog

class PointsDialog(QDialog):
    def __init__(self, parent=None):
        # Class initializations
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Definitions
        rx = QRegularExpression(r"^-?\d+\.\d+(?:,-?\d+\.\d+)*$") # validate floats separated by a comma
        self.validator = QRegularExpressionValidator(rx, self)
        self.error = QErrorMessage(self)

        # tuple of floats to str
        self.to_str = lambda l: ','.join([str(e) for e in l])
        self.to_float = lambda s: tuple([float(e) for e in s.split(',')])

        self.ui.lineEditP.setText(self.to_str(self.parent().P))
        self.ui.lineEditQ.setText(self.to_str(self.parent().Q))

        # Connexions
        self.ui.OKPushButton.clicked.connect(self.validate)
        self.ui.cancelPushButton.clicked.connect(self.reject)

    def validate(self):
        """
        Validate P and Q inputs, propagate member values to the parent or display
        an error message
        """
        # rename for conviniency
        editP = self.ui.lineEditP
        editQ = self.ui.lineEditQ
        # remove spaces from input
        P_text, Q_text = editP.text().replace(" ", ""), editQ.text().replace(" ", "")
        # use the validator, _input and _pos are not used
        P_status, _input, _pos = self.validator.validate(P_text, 0)
        Q_status, _input, _pos = self.validator.validate(Q_text, 0)
        if P_status == self.validator.Acceptable and Q_status == self.validator.Acceptable:
            # Set parent P and Q members
            P = self.to_float(P_text)
            Q = self.to_float(Q_text)
            self.parent().P = P
            self.parent().Q = Q
            # Compute x_step
            N_x = self.parent().nXSpinBox.value() # get N_x value
            # Get the distance between two coordinates (m)
            D = Geodesic.WGS84.Inverse(P[0], P[1], Q[0], Q[1])["s12"]
            self.parent().deltaXMDoubleSpinBox.setValue(D/N_x) # set x_step value in the ui
            self.parent().xMaxKmDoubleSpinBox.setValue(D/1000) # set x_max value in the ui in kilometers
            # Input was validated and propagated to parent, close the dialog
            self.accept()
        else:
            self.error.showMessage("P or Q does not have the correct format.<br>It has to be two floats separated by a comma.<br>Example : <code>34.6553, 34.7654</code>")
