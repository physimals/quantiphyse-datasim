"""
Perfusion simulation Quantiphyse plugin

Structural models, i.e. classes which return lists of different
structures and the corresponding partial volume maps

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption

from .model import Model, Parameter

def get_struc_model(name):
    if name == "fsl1mm":
        return Fsl1mmModel
    elif name == "user":
        return UserPvModel
    else:
        raise QpException("Unknown structural model: %s" % name)

class Fsl1mmModel(Model):
    """
    Structural model using standard FSL 1mm data
    """
    def __init__(self, ivm):
        Model.__init__(self, ivm, "fsl1mm", "FSL 1mm MNI standard data")

    @property
    def structures(self):
        return [
            Parameter("gm", "Grey matter"),
            Parameter("wm", "White matter"),
            Parameter("csf", "CSF"),
        ]

    @classmethod
    def get_structure_map(options, structure, ivm):
        pass

class UserPvModel(Model):
    """
    Structural model where user supplies partial volume maps
    """
    def __init__(self, ivm):
        Model.__init__(self, ivm, "user", "User specified partial volume maps")
        self.gui = OptionBox()
        self.gui.add("GM map", DataOption(self.ivm), key="gm")
        self.gui.add("WM map", DataOption(self.ivm), key="wm")
        self.gui.add("CSF map", DataOption(self.ivm), key="csf")

    @property
    def options(self):
        return self.gui.values()

    @property
    def structures(self):
        return {
            Parameter("gm", "Grey matter", pv=self.gui.option("gm").value),
            Parameter("wm", "White matter", pv=self.gui.option("wm").value),
            Parameter("csf", "CSF", pv=self.gui.option("csf").value),
        }

    @staticmethod
    def get_structure_maps(options, ivm):
        return {
            "gm" : ivm.data[options["gm"]],
            "wm" : ivm.data[options["wm"]],
            "csf" : ivm.data[options["csf"]],
        }
