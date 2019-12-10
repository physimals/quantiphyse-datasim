"""
Perfusion simulation Quantiphyse plugin

Structural models, i.e. classes which return lists of different
structures and the corresponding partial volume maps

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

import numpy as np

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.data import NumpyData, DataGrid
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption
from quantiphyse.utils import QpException, get_plugins

from .model import Model, Parameter

def get_struc_models():
    ret = {}
    for cls in get_plugins("perfsim-struc-models"):
        ret[cls.NAME] = cls
    return ret

class StructureModel(Model):

    def get_simulated_data(self, data_model, param_values):
        """
        Generic function to generate test data from a set of
        partial volume maps, a dictionary of parameter values 
        for each structure
        
        Calls the 'timeseries' method which must be implemented
        by the specific data model
        """
        pv_maps = self.get_structure_maps()

        output_data = None
        for name, pv_map in pv_maps.items():
            print("Generating data for ", name)
            
            struc_values = param_values[name]
            print("struc values", struc_values)

            # Check that there is exactly one parameter value per structure
            single_values = {}
            for k, v in struc_values.items():
                if isinstance(v, (float, int)):
                    v = [v]
                if len(v) != 1:
                    print(v)
                    raise QpException("This structure model cannot handle multiple parameter values in a single structure")
                single_values[k] = v[0]

            timeseries = data_model.get_timeseries(single_values)
            print("timeseries", timeseries)
            if output_data is None:
                output_data = np.zeros(list(pv_map.grid.shape) + [len(timeseries)], dtype=np.float32)
                output_grid = pv_map.grid
                print("Created output data shape", output_data.shape)

            struc_data = pv_map.raw()[..., np.newaxis] * timeseries
            output_data += struc_data

        return NumpyData(output_data, grid=output_grid, name="sim_data")

    def get_structure_maps(self):
        raise NotImplementedError()

def fslimage_to_qpdata(img, name=None, vol=None, region=None):
    """ Convert fsl.data.Image to QpData """
    if not name: name = img.name
    if vol is not None:
        data = img.data[..., vol]
    else:
        data = img.data
    if region is not None:
        data = (data == region).astype(np.int)
    return NumpyData(data, grid=DataGrid(img.shape[:3], img.voxToWorldMat), name=name)

class FslStdStructureModel(StructureModel):
    """
    Structural model using standard FSL structural data
    """
    NAME = "fsl"

    ATLAS_PREFIXES = [
        "MNI",
    ]

    def __init__(self, ivm):
        StructureModel.__init__(self, ivm, "FSL MNI standard data")
        from fsl.data.atlases import AtlasRegistry
        self._registry = AtlasRegistry()
        self._registry.rescanAtlases()
        atlas_names = []
        self._atlases = {}
        for atlas in sorted(self._registry.listAtlases(), key=lambda x: x.name):
            for prefix in self.ATLAS_PREFIXES:
                if atlas.name.startswith(prefix):
                    for pixdim in atlas.pixdims:
                        name = atlas.name + " %.2gx%.2gx%.2g mm" % pixdim
                        self._atlases[name] = (atlas, pixdim)

        self.gui.add("Atlas", ChoiceOption(list(self._atlases.keys())), key="atlas")

    @property
    def structures(self):
        atlas, pixdims = self._atlases[self.gui.option("atlas").value]
        structures = []
        for label in atlas.labels:
            structures.append(Parameter(label.name, label.name))
            print(dir(label))
        return structures

    def get_structure_maps(self):
        atlas, pixdims = self._atlases[self.gui.option("atlas").value]
        structure_maps = {}
        atlas_map = self._registry.loadAtlas(atlas.atlasID, loadSummary=False, resolution=pixdims[0])
        for idx, label in enumerate(atlas.labels):
            structure_maps[label.name] = fslimage_to_qpdata(atlas_map, vol=idx, name=label.name)
        return structure_maps

class UserPvModel(StructureModel):
    """
    Structural model where user supplies partial volume maps
    """
    NAME = "user"

    def __init__(self, ivm):
        StructureModel.__init__(self, ivm, "User specified partial volume maps")
        self.gui = OptionBox()
        self.gui.add("GM map", DataOption(self.ivm, explicit=True), key="gm")
        self.gui.add("WM map", DataOption(self.ivm, explicit=True), key="wm")
        self.gui.add("CSF map", DataOption(self.ivm, explicit=True), key="csf")

    @property
    def structures(self):
        return {
            Parameter("gm", "Grey matter"),
            Parameter("wm", "White matter"),
            Parameter("csf", "CSF"),
        }

    def get_structure_maps(self):
        options = self.options
        return {
            "gm" : self.ivm.data[options["gm"]],
            "wm" : self.ivm.data[options["wm"]],
            "csf" : self.ivm.data[options["csf"]],
        }

class CheckerboardModel(StructureModel):
    """
    Model which builds a checkerboard for up to 3 varying parameters

    This model differs from the usual 'partial volume' type models in that
    it expects multiple values for up to 3 parameters. It builds a 1/2/3D
    grid in which each varying parameter changes along a given dimension.
    There can only be a single 'structure' defined whose name is arbitrary.
    """
    NAME = "checkerboard"

    def __init__(self, ivm):
        StructureModel.__init__(self, ivm, "Checkerboard")
        self.gui = OptionBox()
        self.gui.add("Number of voxels per patch (approx)", NumericOption(minval=1, maxval=1000, default=20, intonly=True), key="voxels-per-patch")

    @property
    def structures(self):
        return {
            Parameter("data", "Sequence of test values"),
        }

    def get_simulated_data(self, data_model, param_values):
        varying_params = []
        if len(param_values) != 1:
            raise QpException("Can only have a single structure in the checkerboard model")
        param_values = list(param_values.values())[0]

        param_values_list = {}
        for param, values in param_values.items():
            if isinstance(values, (int, float)):
                values = [values]
            if len(values) > 1:
                varying_params.append(param)
            param_values_list[param] = values

        num_varying_params = len(varying_params)
        if num_varying_params > 3:
            raise QpException("At most 3 parameters can vary")
        elif num_varying_params == 0:
            # Make it a square for simplicity
            num_varying_params = 2

        voxels_per_patch = self.options.get("voxels-per-patch", 100)
        side_length = int(round(voxels_per_patch ** (1.0 / float(num_varying_params))))
        patch_dims = [side_length] * num_varying_params
        while len(patch_dims) < 3:
            patch_dims += [1,]
        print("Dimensions of patch: ", patch_dims)

        repeats = [[0], [0], [0]]
        checkerboard_dims = []
        for idx, param in enumerate(varying_params):
            num_values = len(param_values_list[param])
            repeats[idx] = range(num_values)
            checkerboard_dims.append(patch_dims[idx] * num_values)
        for idx in range(len(varying_params), 3):
            checkerboard_dims.append(patch_dims[idx])
        print("Dimensions of checkerboard: ", checkerboard_dims)

        output_data = None
        import itertools
        for indexes in itertools.product(*repeats):
            print(indexes)
            patch_values = dict(param_values)
            for idx, param in enumerate(varying_params):
                patch_values[param] = patch_values[param][indexes[idx]]
            
            timeseries = data_model.get_timeseries(patch_values)
            print("timeseries", timeseries)
            if output_data is None:
                output_data = np.zeros(list(checkerboard_dims) + [len(timeseries)], dtype=np.float32)
                print("Created output data shape", output_data.shape)

            slices = []
            for dim_idx, patch_idx in enumerate(indexes):
                dim_length = patch_dims[dim_idx]
                slices.append(slice(patch_idx*dim_length, (patch_idx+1)*dim_length))
            print(slices)
            output_data[slices] = timeseries

        return NumpyData(output_data, grid=DataGrid(checkerboard_dims, np.identity(4)), name="sim_data")
