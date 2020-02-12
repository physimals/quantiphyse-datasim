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
                    self.gui.option(k).value = v
                else:
                    self.nongui_options[k] = v
            except ValueError:
                raise QpException("Invalid value for option '%s': '%s'" % (k, v))
