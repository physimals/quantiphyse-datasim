"""
Perfusion simulation Quantiphyse plugin

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

from .model import Model, Parameter

LOG = logging.getLogger(__name__)

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

class FabberDataModel(DataModel):
    """
    Generates simulated data using Fabber
    """
    NAME = ""
    
    def __init__(self, ivm, title):
        DataModel.__init__(self, ivm, title)
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        self._fab = Fabber(*search_dirs)

        # Include some basic parameters common to many Fabber models
        self.known_params = [
            Parameter(
                "t1",
                "Tissue T1",
                default=1.3,
                units="s",
                struc_defaults={
                    "gm" : 1.3,
                    "wm" : 1.3,
                    "csf" : 1.3,
                },
                aliases=["T_1"],
            ),
            Parameter(
                "t2",
                "Tissue T2",
                default=100,
                units="ms",
                struc_defaults={
                    "gm" : 100,
                    "wm" : 100,
                    "csf" : 100,
                },
                aliases=["T_2"],
            ),
            Parameter(
                "t1b",
                "Blood T1",
                default=1.65,
                units="s",
                struc_defaults={
                    "gm" : 1.65,
                    "wm" : 1.65,
                    "csf" : 1.65,
                },
                aliases=["T1_b", "T_1b"],
            ),
        ]

    @property
    def params(self):
        return [param for param in self.known_params if param.name in self.real_param_names]

    @property
    def real_param_names(self):
        """
        Fabber models often use different names for the same basic parameter (e.g. T1 might be
        named as t1, T1, T_1, ...). This is obviously a pain but we get around it by allowing
        parameters to have 'aliases' and maintaining a mapping from the 'standard' name to the one
        used in the model
        """
        model_params = self._fab.get_model_params(self.fab_options)
        real_param_names = {}
        for param in model_params:
            for known_param in self.known_params:
                if param.lower() == known_param.name.lower() or param in known_param.kwargs.get("aliases", []):
                    real_param_names[known_param.name] = param
                    break
            if param not in real_param_names.values():
                raise ValueError("Unrecognized model parameter: %s" % param)
        return real_param_names

    def get_timeseries(self, param_values):
        LOG.debug("Fabbber options %s", self.fab_options)
        real_param_values = {}
        real_param_names = self.real_param_names
        # Need to correct parameter names from the 'standard' names to those
        # used in the model
        for k, v in param_values.items():
            real_param_values[real_param_names[k]] = v
        ts = self._fab.model_evaluate(self.fab_options, real_param_values, self.nt)
        LOG.debug("Fabbber timeseries %s", ts)
        return ts

    @property
    def fab_options(self):
        raise NotImplementedError()

    @property
    def nt(self):
        raise NotImplementedError()

class AslDataModel(FabberDataModel):
    """
    Generates simulated ASL data using Fabber

    This uses the resting-state ASL model 'aslrest'
    """
    NAME = "asl"
    
    def __init__(self, ivm):
        FabberDataModel.__init__(self, ivm, "Arterial Spin Labelling")

        self.gui.add("Bolus duration", NumericOption(minval=0, maxval=5, default=1.8), key="tau")
        self.gui.add("Labelling", ChoiceOption(["CASL/pCASL", "PASL"], [True, False], default=True), key="casl")
        self.gui.add("PLDs", NumberListOption([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]), key="plds")
        self.gui.add("Arterial component", BoolOption(), key="incart")

        self.known_params += [
            Parameter(
                "ftiss", 
                "CBF",
                default=10.0,
                units="ml/100g/s",
                struc_defaults={
                    "gm" : 50.0,
                    "wm" : 10.0,
                    "csf" : 0.0,
                }
            ),
            Parameter(
                "delttiss",
                "ATT to tissue",
                default=1.3,
                units="s",
                struc_defaults={}
            ),
            Parameter(
                "fblood",
                "Blood CBF",
                default=10.0,
                units="ml/100g/s",
                struc_defaults={}
            ),
            Parameter(
                "deltblood",
                "Transit time to artery",
                default=1.0,
                units="s",
                struc_defaults={}
            ),
        ]

    @property
    def fab_options(self):
        fab_options = {
            "model" : "aslrest",
            "inctiss" : True,
            "incbat" : True,
            "inct1" : True,
        }
        plds = self.options.get("plds", [1.0])
        for idx, pld in enumerate(plds):
            fab_options["pld%i" % (idx+1)] = pld
        fab_options.update(self.options)
        return fab_options

    @property
    def nt(self):
        return len(self.options.get("plds", [1.0]))
           
class DscDataModel(FabberDataModel):
    """
    Generates simulated DSC data
    """
    NAME = "dsc"
    
    def __init__(self, ivm):
        FabberDataModel.__init__(self, ivm, "Dynamic Susceptibility Contrast")

        self.gui.add("Time between volumes (s)", NumericOption(minval=0, maxval=5, default=1.0), key="delt")
        self.gui.add("TE (s)", NumericOption(minval=0, maxval=5, default=1.0), key="te")
        self.gui.add("AIF", NumberListOption(), key="aif")

        self.known_params = [
            Parameter("sig0", "Signal offset", default=100.0),
            Parameter("cbf", "CBF", default=10.0),
        ]

    @property
    def fab_options(self):
        fab_options = {
            "model" : "dsc",
        }
        fab_options.update(self.options)
        return fab_options

    @property
    def nt(self):
        return len(self.options["aif"])
         
class DceDataModel(FabberDataModel):
    """
    Generates simulated DCE data
    """
    NAME = "dce"
    
    def __init__(self, ivm):
        FabberDataModel.__init__(self, ivm, "Dynamic Contrast-Enhanced MRI")
        
        from fabber import Fabber
        search_dirs = get_plugins(key="fabber-dirs")
        self._fab = Fabber(*search_dirs)

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

        self.known_params = [
            Parameter("sig0", "Signal offset", default=100.0),
            Parameter("fp", "Flow, Fp", default=0.5, units="min^-1"),
            Parameter("ps", "Permeability-surface area, PS", default=0.5, units="min^-1"),
            Parameter("ktrans", "Transfer coefficient, Ktrans", default=0.5, units="min^-1"),
            Parameter("ve", "Extracellular volume fraction Ve", default=0.2),
            Parameter("vp", "Vascular plasma volume fraction Vp", default=0.05),
            Parameter("t10", "T1", default=1.0, units="s"),
        ]

    def _aif_changed(self):
        aif_source = self.gui.option("aif").value
        self.gui.set_visible("tinj", aif_source not in ("signal", "conc"))
        self.gui.set_visible("aif-data", aif_source in ("signal", "conc"))
        self.gui.set_visible("nt", aif_source not in ("signal", "conc"))

    def _model_changed(self):
        pass

    @property
    def fab_options(self):
        fab_options = {
            "infer-sig0" : True,
            "infer-fp" : True,
            "infer-ps" : True,
            "infer-t10" : True,
            "infer-delay" : False,
        }
        fab_options.update(self.options)
        if fab_options["model"] == "dce_ETM":
            fab_options["model"] = "dce_tofts"
            fab_options["infer-vp"] = True

        # Transit delay time to include injection time for population AIF
        if "tinj" in fab_options:
            fab_options["delay"] = fab_options["delay"] + fab_options.pop("tinj")

        # Times in minutes and TR in s
        fab_options["delt"] = fab_options["delt"] / 60
        fab_options["delay"] = fab_options["delay"] / 60
        fab_options["tr"] = fab_options["tr"] / 1000

        # NT is not a Fabber option, it is handled separately
        fab_options.pop("nt", None)
        if "aif-data" not in fab_options:
            fab_options["aif-data"] = [1.0]

        return fab_options

    @property
    def nt(self):
        return self.options.get("nt", len(self.options.get("aif-data", [1])))
         
         