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
from quantiphyse.utils import get_plugins

from .model import Model, Parameter

def get_data_model(name):
    if name == "asl":
        return AslDataModel
    else:
        raise QpException("Unknown data model: %s" % name)
        
class AslDataModel(Model):
    """
    Generates simulated ASL data
    """
    def __init__(self, ivm):
        Model.__init__(self, ivm, "asl", "Arterial Spin Labelling")
        self.gui = OptionBox()
        self.gui.add("PLDs", NumberListOption(), key="plds")

    @property
    def params(self):
        return [
            Parameter("ftiss", "CBF", default=10.0, units="ml/100g/s"),
            Parameter("delttiss", "ATT", default=1.3, units="s"),
        ]

    @property
    def options(self):
        print("options: ", self.gui.values())
        return self.gui.values()

    @staticmethod
    def generate_data(options, param_values, pv_maps):
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        fab = Fabber(*search_dirs)

        plds = options["plds"]
        fab_options = {
            "model" : "aslrest",
            "casl" : True,
            "tau" : 1.8,
            "inctiss" : True,
            "incbat" : True,
        }
        for idx, pld in enumerate(plds):
            fab_options["pld%i" % (idx+1)] = pld

        output_data = None
        for name, pv_map in pv_maps.items():
            print("Generating data for ", name)
            if output_data is None:
                output_data = np.zeros(list(pv_map.grid.shape) + [len(plds)], dtype=np.float32)
                output_grid = pv_map.grid
                print("Created output data shape", output_data.shape)

            struc_values = param_values[name]
            print("struc values", struc_values)
            timeseries = fab.model_evaluate(fab_options, struc_values, len(plds))
            print("timeseries", timeseries)
            struc_data = np.zeros(output_data.shape, dtype=np.float32)
            struc_data = pv_map.raw()[..., np.newaxis] * timeseries
            output_data += struc_data

        return NumpyData(output_data, name="sim_data", grid=output_grid)