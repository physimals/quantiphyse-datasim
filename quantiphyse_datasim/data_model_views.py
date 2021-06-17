"""
Datausion simulation Quantiphyse plugin

Data models, i.e. classes which generated simulated data for different
types of imaging

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

import logging

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

import numpy as np

from fabber import Fabber

from quantiphyse.data import NumpyData
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption
from quantiphyse.utils import get_plugins, QpException

from .data_models import *

LOG = logging.getLogger(__name__)

class SpinEchoDataModelView:
    def __init__(self, ivm):
        self.model = SpinEchoDataModel(ivm)
        self.gui = OptionBox()
        self.gui.add("TR (s)", NumericOption(minval=0, maxval=10, default=4.8), key="tr")
        self.gui.add("TE (ms)", NumericOption(minval=0, maxval=1000, default=0), key="te")
        self.gui.add("M0", NumericOption(minval=0, maxval=10000, default=1000), key="m0")
        self.gui.sig_changed.connect(self._update_options)
        self._update_options()

    def _update_options(self):
        self.model.options.update(self.gui.values())

class AslDataModelView:
    def __init__(self, ivm):
        self.model = AslDataModel(ivm)
        self.gui = OptionBox()
        self.gui.add("Bolus duration", NumericOption(minval=0, maxval=5, default=1.8), key="tau")
        self.gui.add("Labelling", ChoiceOption(["CASL/pCASL", "PASL"], [True, False], default=True), key="casl")
        self.gui.add("PLDs", NumberListOption([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]), key="plds")
        self.gui.add("Time per slice (ms)", NumericOption(minval=0, maxval=1000, default=0, intonly=True), key="slicedt")
        self.gui.add("Data format", ChoiceOption(["Differenced data", "Label/Control pairs"], ["diff", "tc"]), key="iaf")
        self.gui.add("Repeats", NumericOption(minval=1, maxval=100, default=1, intonly=True), key="repeats")
        self.gui.add("Group by", ChoiceOption(["PLDs", "Repeats"], ["tis", "rpt"]), key="ibf")
        self.gui.add("Inversion efficiency", NumericOption(minval=0.5, maxval=1.0, default=0.85), key="alpha")
        self.gui.add("M0", NumericOption(minval=0, maxval=2000, default=1000), key="m0")
        self.gui.add("TR (s)", NumericOption(minval=0, maxval=10, default=4), key="tr")
        self.gui.add("TE (ms)", NumericOption(minval=0, maxval=1000, default=13), key="te")
        self.gui.add("Tissue/arterial partition coefficient", NumericOption(minval=0, maxval=1, default=0.9), key="pct")
        #self.gui.add("Arterial component", BoolOption(), key="incart")
        self.gui.sig_changed.connect(self._update_options)
        self._update_options()

    def _update_options(self):
        self.model.options.update(self.gui.values())

class DscDataModelView:
    def __init__(self, ivm):
        self.model = DscDataModel(ivm)
        self.gui = OptionBox()
        self.gui.add("Time between volumes (s)", NumericOption(minval=0, maxval=5, default=1.0), key="delt")
        self.gui.add("TE (s)", NumericOption(minval=0, maxval=5, default=1.0), key="te")
        self.gui.add("AIF", NumberListOption(), key="aif")
        self.gui.sig_changed.connect(self._update_options)
        self._update_options()

    def _update_options(self):
        self.model.options.update(self.gui.values())
         
class DceDataModelView:
    def __init__(self, ivm):
        self.model = DceDataModel(ivm)
        self.gui = OptionBox()
        self.gui.add("Model", ChoiceOption(["Standard Tofts model",
                                             "Extended Tofts model (ETM)",
                                             "2 Compartment exchange model",
                                             "Compartmental Tissue Update (CTU) model",
                                             "Adiabatic Approximation to Tissue Homogeneity (AATH) Model"],
                                            ["dce_tofts",
                                             "dce_ETM",
                                             "dce_2CXM",
                                             "dce_CTU",
                                             "dce_AATH"]), key="model")
        self.gui.add("Contrast agent R1 relaxivity (l/mmol s)", NumericOption(minval=0, maxval=10, default=3.7), key="r1")
        self.gui.add("Flip angle (\N{DEGREE SIGN})", NumericOption(minval=0, maxval=90, default=12), key="fa")
        self.gui.add("TR (ms)", NumericOption(minval=0, maxval=10, default=4.108), key="tr")
        self.gui.add("Time between volumes (s)", NumericOption(minval=0, maxval=30, default=12), key="delt")
        self.gui.add("AIF", ChoiceOption(["Population (Orton 2008)", "Population (Parker)", "Measured DCE signal", "Measured concentration curve"], ["orton", "parker", "signal", "conc"]), key="aif")
        self.gui.add("Number of volumes", NumericOption(minval=0, maxval=100, default=20, intonly=True), key="nt")
        self.gui.add("Bolus injection time (s)", NumericOption(minval=0, maxval=60, default=30), key="tinj")
        self.gui.add("AIF data values", NumberListOption([0, ]), key="aif-data")
        self.gui.add("Arterial transit time (s)", NumericOption(minval=0, maxval=1.0, default=0), key="delay")
        self.gui.option("model").sig_changed.connect(self._model_changed)
        self.gui.option("aif").sig_changed.connect(self._aif_changed)
        self._aif_changed()
        self._model_changed()
        self.gui.sig_changed.connect(self._update_options)
        self._update_options()

    def _aif_changed(self):
        aif_source = self.gui.option("aif").value
        self.gui.set_visible("tinj", aif_source not in ("signal", "conc"))
        self.gui.set_visible("aif-data", aif_source in ("signal", "conc"))
        self.gui.set_visible("nt", aif_source not in ("signal", "conc"))

    def _model_changed(self):
        pass

    def _update_options(self):
        self.model.options.update(self.gui.values())
