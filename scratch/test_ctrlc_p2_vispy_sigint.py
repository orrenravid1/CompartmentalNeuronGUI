"""
Test P2: does vispy_app.run() / Qt swallow KeyboardInterrupt?

Run: python scratch/test_ctrlc_p2_vispy_sigint.py
Then Ctrl+C.
Expected if P2 is bug: no KeyboardInterrupt raised in Python — app keeps running.
Expected if NOT: KeyboardInterrupt propagates out of vispy_app.run().
"""
import sys
from PyQt6 import QtWidgets
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

qapp = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
window.setWindowTitle("P2 test — Ctrl+C propagation")
window.resize(400, 200)
window.show()

try:
    vispy_app.run()
    print("vispy_app.run() returned normally")
except KeyboardInterrupt:
    print("KeyboardInterrupt propagated out of vispy_app.run() — P2 NOT the bug")
