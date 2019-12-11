"""
Perfusion simulation Quantiphyse plugin

Data models, i.e. classes which generated simulated data for different
types of imaging

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
    Implements a model for producing simulated data

    A data model has the ability to generate timeseries data from a dictionary
    of parameter values.
    """

    def get_timeseries(self, param_values):
        raise NotImplementedError()

class AslDataModel(DataModel):
    """
    Generates simulated ASL data using Fabber

    This uses the resting-state ASL model 'aslrest'
    """
    NAME = "asl"
    
    def __init__(self, ivm):
        DataModel.__init__(self, ivm, "Arterial Spin Labelling")
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        self._fab = Fabber(*search_dirs, debug=True)

        self.gui.add("Bolus duration", NumericOption(minval=0, maxval=5, default=1.8), key="tau")
        self.gui.add("Labelling", ChoiceOption(["CASL/pCASL", "PASL"], [True, False], default=True), key="casl")
        self.gui.add("PLDs", NumberListOption([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]), key="plds")

    @property
    def params(self):
        return [
            Parameter("ftiss", "CBF", default=10.0, units="ml/100g/s"),
            Parameter("delttiss", "ATT", default=1.3, units="s"),
        ]

    @property
    def fab_options(self):
        fab_options = {
            "model" : "aslrest",
            "inctiss" : True,
            "incbat" : True,
        }
        plds = self.options.get("plds", [1.0])
        for idx, pld in enumerate(plds):
            fab_options["pld%i" % (idx+1)] = pld
        fab_options.update(self.options)
        return fab_options

    def get_timeseries(self, param_values):
        nt = len(self.options.get("plds", [1.0]))
        return self._fab.model_evaluate(self.fab_options, param_values, nt)
           
class DscDataModel(DataModel):
    """
    Generates simulated DSC data
    """
    NAME = "dsc"
    
    def __init__(self, ivm):
        Model.__init__(self, ivm, "Dynamic Susceptibility Contrast")

        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        self._fab = Fabber(*search_dirs, debug=True)

        self.gui = OptionBox()
        self.gui.add("Time between volumes (s)", NumericOption(minval=0, maxval=5, default=1.0), key="delt")
        self.gui.add("TE (s)", NumericOption(minval=0, maxval=5, default=1.0), key="te")
        self.gui.add("AIF", NumberListOption(), key="aif")

    @property
    def params(self):
        model_params = self._fab.get_model_params(self.fab_options)
        known_params = [
            Parameter("sig0", "Signal offset", default=100.0),
            Parameter("cbf", "CBF", default=10.0),
        ]
        return [param for param in known_params if param.name in model_params]

    @property
    def fab_options(self):
        fab_options = {
            "model" : "dsc",
        }
        fab_options.update(self.options)
        print(fab_options)
        # Model expects time resolution in minutes
        fab_options["delt"] = float(fab_options["delt"]) / 60.0
        return fab_options

    def get_timeseries(self, param_values):
        # AIF defines number of time points
        nt = len(self.options["aif"])
        return self._fab.model_evaluate(self.fab_options, param_values, nt)
