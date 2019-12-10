"""
Perfusion simulation Quantiphyse plugin

Generic classes for data and structural models

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

from quantiphyse.gui.options import OptionBox
from quantiphyse.utils import QpException

class Parameter:
    def __init__(self, name, display_name, **kwargs):
        self.name = name
        self.display_name = display_name
        self.kwargs = kwargs

class Model:
    """
    Provides ROIs for named structures in which perfusion will be modelled
    """
    def __init__(self, ivm, display_name):
        self.ivm = ivm
        self.display_name = display_name
        self.gui = OptionBox()

    @property
    def options(self):
        return self.gui.values()

    @options.setter
    def options(self, options):
        for k, v in options.items():
            try:
                self.gui.option(k).value = v
            except:
                raise QpException("Invalid value for option '%s': '%s'" % (k, v))