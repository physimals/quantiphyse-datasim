"""
Perfusion simulation Quantiphyse plugin

Data models, i.e. classes which generated simulated data for different
modalities

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

import numpy as np

from fabber import Fabber

from quantiphyse.data import NumpyData
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption
from quantiphyse.utils import get_plugins, QpException

from .model import Model, Parameter

def get_data_models():
    ret = {}
    for cls in get_plugins("perfsim-data-models"):
        ret[cls.NAME] = cls
    return ret

class DataModel(Model):
    """
    """

    def get_timeseries(self, param_values):
        raise NotImplementedError()

class AslDataModel(DataModel):
    """
    Generates simulated ASL data
    """
    NAME = "asl"
    
    def __init__(self, ivm):
        DataModel.__init__(self, ivm, "Arterial Spin Labelling")
        self.gui.add("Bolus duration", NumericOption(minval=0, maxval=5, default=1.8), key="tau")
        self.gui.add("PLDs", NumberListOption(), key="plds")

    @property
    def params(self):
        return [
            Parameter("ftiss", "CBF", default=10.0, units="ml/100g/s"),
            Parameter("delttiss", "ATT", default=1.3, units="s"),
        ]

    def get_timeseries(self, param_values):
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        fab = Fabber(*search_dirs)

        plds = self.options.get("plds", [1.0])
        fab_options = {
            "model" : "aslrest",
            "casl" : True,
            "inctiss" : True,
            "incbat" : True,
        }
        for idx, pld in enumerate(plds):
            fab_options["pld%i" % (idx+1)] = pld
        fab_options.update(self.options)

        print(fab_options, param_values)
        return fab.model_evaluate(fab_options, param_values, len(plds))
           
class DscDataModel(DataModel):
    """
    Generates simulated DSC data
    """
    NAME = "dsc"
    
    def __init__(self, ivm):
        Model.__init__(self, ivm, "Dynamic Susceptibility Contrast")
        self.gui = OptionBox()
        self.gui.add("Time between volumes (s)", NumericOption(minval=0, maxval=5, default=1.0), key="tr")
        self.gui.add("TE (s)", NumericOption(minval=0, maxval=5, default=1.0), key="te")
        self.gui.add("AIF", NumberListOption(), key="aif")

    @property
    def params(self):
        return [
        ]

    @property
    def options(self):
        print("options: ", self.gui.values())
        return self.gui.values()

    @staticmethod
    def get_timeseries(options, param_values):
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        fab = Fabber(*search_dirs)

        fab_options = {
            "model" : "dsc",
        }
        fab_options.update(options)

        return fab.model_evaluate(fab_options, param_values, len(plds))
           