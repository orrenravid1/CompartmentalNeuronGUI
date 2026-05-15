"""
Test possibility 1: QApplication GC'd if ref not stored.
Does vispy_app.run() exit immediately when QApplication has no stored ref?

Run: python scratch/test_p1_qapp_gc.py
Expected if P1 is the bug: window opens but closes instantly (or never appears).
Expected if P1 is NOT the bug: window stays open.
"""
import sys
from PyQt6 import QtWidgets
from vispy import app as vispy_app, use

use(app="pyqt6", gl="gl+")

# No stored ref — same as current VispyFrontendHost.start()
if QtWidgets.QApplication.instance() is None:
    QtWidgets.QApplication(sys.argv)

window = QtWidgets.QMainWindow()
window.setWindowTitle("P1 test — no stored qapp ref")
window.resize(400, 200)
window.show()

print("Calling vispy_app.run() — should block until window closed")
vispy_app.run()
print("vispy_app.run() returned")
