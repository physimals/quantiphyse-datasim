"""
Perfusion simulation Quantiphyse plugin

Generic classes for data and structural models

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

class Parameter:
    def __init__(self, name, display_name, **kwargs):
        self.name = name
        self.display_name = display_name
        self.kwargs = kwargs

class Model:
    """
    Provides ROIs for named structures in which perfusion will be modelled
    """
    def __init__(self, ivm, name, display_name):
        self.ivm = ivm
        self.name = name
        self.display_name = display_name
        self.gui = None

    @property
    def options(self):
        return {}
