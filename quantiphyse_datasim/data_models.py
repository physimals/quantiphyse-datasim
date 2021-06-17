"""
Data simulation Quantiphyse plugin

Data models, i.e. classes which generated simulated data for different
types of imaging

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

import logging

import numpy as np

from fabber import Fabber

from quantiphyse.data import NumpyData
from quantiphyse.utils import get_plugins, QpException

from .model import Model, Parameter

LOG = logging.getLogger(__name__)

def get_data_models():
    ret = {}
    for cls in get_plugins("datasim-data-models"):
        ret[cls.NAME] = cls
    return ret

class DataModel(Model):
    """
    Implements a model for producing simulated data

    A data model has the ability to generate timeseries data from a dictionary
    of parameter values.
    """

    def __init__(self, ivm, title):
        Model.__init__(self, ivm, title)

        # Include some basic parameters common to many models
        self.known_params = [
            Parameter(
                "t1",
                "Tissue T1",
                default=1.3,
                units="s",
                struc_defaults={
                    "gm" : 1.3,
                    "wm" : 1.1,
                    "csf" : 4.3,
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
                    "wm" : 50,
                    "csf" : 750,
                },
                aliases=["T_2"],
            ),
            Parameter(
                "pc",
                "Partition coefficient",
                default=0.9,
                units="",
                struc_defaults={
                    "gm" : 0.98,
                    "wm" : 0.82,
                    "csf" : 1.15,
                },
                aliases=["lambda"],
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

    def get_timeseries(self, param_values, shape=None):
        """
        Get the data model time series for a set of parameter values

        :param param_values: Mapping from parameter name to value
        :param shape: If specified, 3D output shape for the data being constructed.
                      Data models may return either a 1D timeseries to 
                      be used for all voxels, or a 4D voxelwise map. If shape is None
                      a 1D timeseries must be returned
        :return: 1D or 4D np.array containing the timeseries
        """
        raise NotImplementedError()

class SpinEchoDataModel(Model):
    """
    Implements a model for producing simulated spin-echo data
    """
    NAME = "spin_echo"

    def __init__(self, ivm):
        DataModel.__init__(self, ivm, "Spin Echo")

    def get_timeseries(self, param_values, shape=None):
        tr = self.options["tr"]
        te = self.options["te"]
        m0 = self.options.get("m0", 1000)
        t1 = param_values.get("t1", 1.3)
        t2 = param_values.get("t2", 100)
        pc = param_values.get("pc", 1)
        # FIXME proton density?
        return np.array([m0 * pc * (1-np.exp(-tr/t1)) * np.exp(-te/t2)])

    @property
    def params(self):
        return [param for param in self.known_params if param.name in ("t1", "t2", "pc")]

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
            # Note that we do not worry if Fabber is returning parameters that are not in our list of
            # known parameters. The data model is responsible for adding in values for these and if it
            # does not an error will occur on evaluation of the model. For example static tissue signal
            # in ASL may be derived from tissue properties (T1, T2 etc)
            for known_param in self.known_params:
                if param.lower() == known_param.name.lower() or param in known_param.kwargs.get("aliases", []):
                    real_param_names[known_param.name] = param
                    break
        return real_param_names

    def get_timeseries(self, param_values, shape=None):
        LOG.debug("Fabbber options %s", self.fab_options)
        real_param_values = {}
        real_param_names = self.real_param_names
        # Need to correct parameter names from the 'standard' names to those
        # used in the model
        for k, v in param_values.items():
            real_param_values[real_param_names[k]] = v
        ts = self._fab.model_evaluate(self.fab_options, real_param_values, self.nt)
        LOG.debug("Fabbber timeseries %s", ts)
        return np.array(ts)

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
        self.slicet = 0
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
                struc_defaults={
                    "gm" : 1.3,
                    "wm" : 1.6,
                    "csf" : 1.3,
                },
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

    def get_timeseries(self, param_values, shape=None):
        if shape is not None and self.options["slicedt"] != 0:
            ret = np.zeros(list(shape) + [self.nt], dtype=np.float32)
            for z in range(shape[2]):
                self.slicet = z*float(self.options["slicedt"])/1000
                ret[:, :, z, :] = self._get_slice_timeseries(param_values)
            self.slicet = 0
        else:
            self.slicet = 0
            ret = self._get_slice_timeseries(param_values)
        return ret

    def _get_slice_timeseries(self, param_values):
        ts = FabberDataModel.get_timeseries(self, param_values)
        ts = self.options["m0"] * self.options["alpha"] * ts / 6000
        nrpt = self.options["repeats"]

        # For tag control pairs calculate static tissue signal at each
        # PLD and add to Fabber timeseries. Note that fabber by default
        # uses zero for control images and by default groups by repeats
        if self.options["iaf"] == "tc":
            tis = np.array(self.options["plds"]) + self.options["tau"] + self.slicet
            tr = self.options["tr"]
            te = self.options["te"]
            t1 = param_values.get("t1", 1.3)
            t2 = param_values.get("t2", 100)
            t1b = param_values.get("t1b", 1.65)
            stattiss = np.tile(self.options["pct"] * self.options["m0"] * (1-np.exp(-tr/t1)) * np.exp(-te/t2), len(tis))
            tc = []
            for idx, sig in enumerate(ts):
                tc.append(stattiss[int(idx/(2*nrpt))] - sig)
            ts = np.array(tc)

        ret = np.tile(ts, nrpt)

        # If we are grouping by TIs we need to reorder the timeseries so all the
        # full set of TIs is together and then repeated
        if self.options["ibf"] == "tis":
            reordered = []
            npld = len(self.options["plds"])
            ntc = 1 if self.options["iaf"] == "diff" else 2
            for pld in range(npld):
                for rpt in range(nrpt):
                    for tc in range(ntc):
                        reordered.append(ret[pld*nrpt*ntc + rpt*ntc + tc])
            ret = np.array(reordered)

        return ret

    @property
    def fab_options(self):
        fab_options = {
            "model" : "aslrest",
            "inctiss" : True,
            "incbat" : True,
            "inct1" : True,
        }
        plds = self.options["plds"]
        for idx, pld in enumerate(plds):
            fab_options["pld%i" % (idx+1)] = pld + self.slicet
        fab_options.update(self.options)

        for opt in ("m0", "alpha", "pct", "slicedt"):
            fab_options.pop(opt, None)

        return fab_options

    @property
    def nt(self):
        diff = self.options["iaf"] == "diff"
        return len(self.options["plds"]) * (1 if diff else 2) * self.options["repeats"]
           
class DscDataModel(FabberDataModel):
    """
    Generates simulated DSC data
    """
    NAME = "dsc"
    
    def __init__(self, ivm):
        FabberDataModel.__init__(self, ivm, "Dynamic Susceptibility Contrast")

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

        self.known_params = [
            Parameter("sig0", "Signal offset", default=100.0),
            Parameter("fp", "Flow, Fp", default=0.5, units="min^-1"),
            Parameter("ps", "Permeability-surface area, PS", default=0.5, units="min^-1"),
            Parameter("ktrans", "Transfer coefficient, Ktrans", default=0.5, units="min^-1"),
            Parameter("ve", "Extracellular volume fraction Ve", default=0.2),
            Parameter("vp", "Vascular plasma volume fraction Vp", default=0.05),
            Parameter("t10", "T1", default=1.0, units="s"),
        ]

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
         
         