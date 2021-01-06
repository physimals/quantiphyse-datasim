"""
Perfusion simulation Quantiphyse plugin

Generic classes for data and structural models

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

from quantiphyse.gui.options import OptionBox
from quantiphyse.utils import QpException

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

class Parameter:
    def __init__(self, name, display_name, **kwargs):
        self.name = name
        self.display_name = display_name
        self.kwargs = kwargs

class Model(QtCore.QObject):
    """
    """
    sig_changed = QtCore.Signal()

    def __init__(self, ivm, display_name):
        QtCore.QObject.__init__(self)
        self.ivm = ivm
        self.display_name = display_name
        self.gui = OptionBox()
        self.gui.sig_changed.connect(self.sig_changed.emit)
        self.nongui_options = {}

    @property
    def options(self):
        opts = self.gui.values()
        opts.update(self.nongui_options)
        return opts
        
    @options.setter
    def options(self, options):
        self.nongui_options = {}
        for k, v in options.items():
            try:
                if self.gui.has_option(k):
                    if not self.gui.option(k).isEnabled():
                        self.gui.set_checked(k, True)
                    self.gui.option(k).value = v
                else:
                    self.nongui_options[k] = v
            except ValueError:
                raise QpException("Invalid value for option '%s': '%s'" % (k, v))
