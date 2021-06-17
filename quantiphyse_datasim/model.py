"""
Data simulation Quantiphyse plugin

Generic classes for data and structural models

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

from quantiphyse.data.qpdata import Metadata
from quantiphyse.utils import QpException, LogSource

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

class Parameter:
    def __init__(self, name, display_name=None, **kwargs):
        self.name = name
        if display_name is None:
            display_name = name
        self.display_name = display_name
        self.kwargs = kwargs

class Model(LogSource):
    """
    """
    def __init__(self, ivm, display_name, **kwargs):
        LogSource.__init__(self)
        self._ivm = ivm
        self.display_name = display_name
        self.options = Metadata()
