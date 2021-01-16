"""
Perfusion simulation Quantiphyse plugin

Structural models, i.e. classes which return lists of different
structures and the corresponding partial volume maps

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

import time

import numpy as np

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.data import NumpyData, DataGrid, ImageVolumeManagement
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption, RunButton
from quantiphyse.utils import QpException, get_plugins
from quantiphyse.processes import Process

from .model import Model, Parameter

def get_struc_models():
    ret = {}
    for cls in get_plugins("perfsim-struc-models"):
        ret[cls.NAME] = cls
    return ret

class StructureModel(Model):
    """
    Base class for a structure model
    
    The ``structures`` property must be defined as a sequence of Parameter objects
    """

    @property
    def structures(self):
        raise NotImplementedError()

    def get_simulated_data(self, data_model, param_values, output_param_maps=False):
        """
        Generate simulated data for a given data model and parameter values

        :param data_model: DataModel which implements the ``timeseries`` method
        :param param_values: Mapping from structure name to sequence of parameter values
        :output_param_maps: If True, also output QpData containing maps of the 
                            parameter values used to generate simulated data in each voxel

        @return If ``output_param_maps`` is ``False``, QpData containing simulated timeseries data
                If ``output_param_maps`` is ``True``, tuple of simulated timeseries data and mapping
                of param name to parameter value map
        """
        raise NotImplementedError()

class PartialVolumeStructureModel(Model):
    """
    Structure model which defines the structure as a set of partial volume maps
    """

    @property
    def structure_maps(self):
        """
        :return: Mapping from name to a QpData instance containing partial volume
                 maps (range 0-1) for each known structure
        """
        raise NotImplementedError()

    def get_simulated_data(self, data_model, param_values, output_param_maps=False):
        """
        Generic implementation to generate test data from a set of partial volume maps
        """
        if len(self.structure_maps) == 0:
            raise QpException("No structures defined")

        # First check the PV maps - they should all be in the range 0-1 and not sum to
        # more than 1 in any voxel
        sum_map = None
        for name, pv_map in self.structure_maps.items():
            pv_data = pv_map.raw()
            if not np.all(pv_data >= 0):
                raise QpException("Partial volume map contained negative values: %s" % name)
            if not np.all(pv_data <= 1):
                raise QpException("Partial volume map contained values > 1: %s" % name)
            if sum_map is None:
                sum_map = np.zeros(pv_data.shape, dtype=np.float32)
            sum_map += pv_data

        if sum_map is not None and not np.all(sum_map <= 1):
            raise QpException("Partial volume maps sum to > 1 in at least one voxel")

        output_data = None 
        for name, pv_map in self.structure_maps.items():
            struc_values = param_values[name]

            # Check that there is exactly one parameter value per structure
            single_values = {}
            for k, v in struc_values.items():
                if isinstance(v, (float, int)):
                    v = [v]
                if len(v) != 1:
                    raise QpException("This structure model cannot handle multiple parameter values in a single structure")
                single_values[k] = v[0]

            timeseries = data_model.get_timeseries(single_values)
            if output_data is None:
                output_data = np.zeros(list(pv_map.grid.shape) + [len(timeseries)], dtype=np.float32)
                output_grid = pv_map.grid
                if output_param_maps:
                    param_maps = {}
                    for param in struc_values:
                        param_maps[param] = np.zeros(pv_map.grid.shape, dtype=np.float32)

            struc_data = pv_map.raw()[..., np.newaxis] * timeseries
            output_data += struc_data

            if output_param_maps:
                for param, value in single_values.items():
                    param_maps[param] += pv_map.raw() * value

        sim_data = NumpyData(output_data, grid=output_grid, name="sim_data")

        if output_param_maps:
            for param in struc_values: 
                param_maps[param] = NumpyData(param_maps[param], grid=output_grid, name=param)
            return sim_data, param_maps
        else:
            return sim_data

class AddEmbeddingDialog(QtGui.QDialog):
    """
    Dialog box enabling one item to be chosen from a list
    """

    def __init__(self, parent, existing_strucs):
        super(AddEmbeddingDialog, self).__init__(parent)
        self.sel_text = None
        self.sel_data = None
        self.existing_names = [struc.name for struc in existing_strucs]

        self.setWindowTitle("Add embedding")
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)

        self._opts = OptionBox()
        name = self._opts.add("Name of embedded structure", TextOption(), key="name")
        name.textChanged.connect(self._name_changed)
        self._opts.add("Structure type", ChoiceOption(["Additional PVE", "Embedding", "Activation mask"], return_values=["add", "embed", "act"]), key="type")
        self._opts.add("Parent structure", ChoiceOption([s.display_name for s in existing_strucs], [s.name for s in existing_strucs]), key="parent")
        vbox.addWidget(self._opts)

        self.button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QtGui.QDialogButtonBox.Ok).setEnabled(False)
        vbox.addWidget(self.button_box)

    def _name_changed(self):
        accept = self.name != "" and self.name not in self.existing_names
        self.button_box.button(QtGui.QDialogButtonBox.Ok).setEnabled(accept)

    @property
    def name(self):
        return self._opts.option("name").value

    @property
    def struc_type(self):
        return self._opts.option("type").value

    @property
    def parent(self):
        return self._option.option("parent").value

class UserPvModel(PartialVolumeStructureModel):
    """
    Structural model where user supplies partial volume maps

    Three default tissue types are defined: Grey matter (GM), White matter (WM) and
    CSF. A partial volume map can be specified for each of these (but they are not
    all compulsary)

    The model also supports additional user-defined structures which come in three types:

     - Additional partial volume maps. These are simply added to the existing default 
       structures. The total sum in any voxel must not exceed 1
     - Embeddings. These are added to the existing default maps, which are scaled to
       1 - the embedding partial volume. For example where the embedding partial volume
       is 1 other structures have zero partial volume. Where the embedding partial volume
       is 0.8, all other structures are scaled by multiplying by 0.2.
     - Activation mask. These are ROIs that cause a specific tissue type to replaced by
       a different set of parameter values within the ROI. For example a GM activation
       mask will replace the 'usual' GM parameters with another set of parameters within
       the mask. This mask can be used to simulate activation of a particular region, e.g.
       derived from a brain atlas.
    """
    NAME = "user"

    def __init__(self, ivm):
        PartialVolumeStructureModel.__init__(self, ivm, "User specified partial volume maps", title="Structure maps")
        self.default_strucs = [
            Parameter("gm", "Grey matter"),
            Parameter("wm", "White matter"),
            Parameter("csf", "CSF"),
        ]
        self.nongui_options = {"additional" : []}
        self._refresh_opts()

    def _refresh_opts(self):
        options = self.options
        self.gui.clear()
        for struc in self.default_strucs:
            self.gui.add("%s map" % struc.name.upper(), DataOption(self.ivm, explicit=True), checked=True, enabled=struc.name in options, key=struc.name)
        for struc in self.nongui_options.get("additional", []):
            self.gui.add("%s map" % struc.name, DataOption(self.ivm, explicit=True), key=struc.name)
            
        self.gui.add(None, RunButton("Add user-defined structure", callback=self._add_embedding), key="add_embedding")

    @property
    def options(self):
        opts = {
            "pvmaps" : self.gui.values()
        }
        opts.update(self.nongui_options)
        return opts

    @options.setter
    def options(self, options):
        self.nongui_options["additional"] = options.pop("additional", {})
        self._refresh_opts()

        for k, v in options.pop("pvmaps", {}).items():
            try:
                if self.gui.has_option(k):
                    if not self.gui.option(k).isEnabled():
                        self.gui.set_checked(k, True)
                    self.gui.option(k).value = v
                else:
                    raise QpException("PV map '%s' given for unrecognized structure '%s'" % (v, k))
            except ValueError:
                raise QpException("Invalid value for option '%s': '%s'" % (k, v))

    def _add_embedding(self):
        dialog = AddEmbeddingDialog(self.gui, self.default_strucs)
        try:
            accept = dialog.exec_()
        except:
            import traceback
            traceback.print_exc()
        if accept:
            self.nongui_options.get("additional", []).append(Parameter(dialog.name, "Embedding"))
            options = self.options
            self._refresh_opts()
            self.options = options

    @property
    def structures(self):
        ret = [struc for struc in self.default_strucs if struc.name in self.options["pvmaps"]] + self.nongui_options.get("additional", [])
        return ret

    @property
    def structure_maps(self):
        options = self.options
        pvmaps = options.get("pvmaps", {})
        try:
            ret = {}
            total_pv = None
            for struc in self.default_strucs:
                if struc.name in pvmaps:
                    ret[struc.name] = self.ivm.data[pvmaps[struc.name]]
                    if total_pv is None:
                        total_pv = np.zeros(ret[struc.name].grid.shape, dtype=np.float32)
                    total_pv += ret[struc.name].raw()
            
            # Additional - we need to downweight existing structure PVs so they all sum to 1 at most
            for struc in self.nongui_options.get("additional", []):
                data = self.ivm.data[pvmaps[struc.name]]
                reweighting = 1-data.raw()
                for name, qpdata in ret.items():
                    qpdata = NumpyData(qpdata.raw() * reweighting, grid=qpdata.grid, name=name)
                    ret[name] = qpdata
                ret[struc.name] = data

            return ret
        except KeyError as exc:
            raise
            #raise QpException("No structure map defined: %s" % str(exc.args[0]))

    def _reweight(self, strucs, to_pv):
        pass

class FastStructureModel(PartialVolumeStructureModel):
    """
    Structural model which derives partial volume maps from a FAST segmentation
    """
    NAME = "fast"

    def __init__(self, ivm):
        StructureModel.__init__(self, ivm, "Partial volume maps from a FAST segmentation")
        self.gui.add("Structural image (brain extracted)", DataOption(self.ivm, explicit=True), key="struc")
        self.gui.add("Image type", ChoiceOption(["T1 weighted", "T2 weighted", "Proton Density"], return_values=[1, 2, 3]), key="type")
        
    @property
    def structures(self):
        return [
            Parameter("gm", "Grey matter"),
            Parameter("wm", "White matter"),
            Parameter("csf", "CSF"),
        ]

    @property
    def structure_maps(self):
        processes = get_plugins("processes", "FastProcess")
        if len(processes) != 1:
            raise QpException("Can't identify Fast process")
        
        struc = self.options.get("struc", None)
        if struc not in self.ivm.data:
            raise QpException("Structural image not loaded: %s" % struc)
        
        qpdata = self.ivm.data[struc]
        ivm = ImageVolumeManagement()
        ivm.add(qpdata)
        process = processes[0](ivm)
        fast_options = {
            "data" : qpdata.name,
            "class" : 3,
            "type" : self.options["type"],
            "output-pve" : True,
            "output-pveseg" : False,
        }
        process.execute(fast_options)
        while process.status == Process.RUNNING:
            time.sleep(1)

        if process.status == Process.FAILED:
            raise process.exception

        # FIXME hack
        process._complete()

        options = self.options
        return {
            "gm" : ivm.data["%s_pve_1" % qpdata.name],
            "wm" : ivm.data["%s_pve_2" % qpdata.name],
            "csf" : ivm.data["%s_pve_0" % qpdata.name],
        }

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

class FslStdStructureModel(PartialVolumeStructureModel):
    """
    Structural model using standard FSL structural data

    FIXME not functional at present - not clear that FSL supplies relevant
    segmentation data out of the box!
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
        return structures

    @property
    def structure_maps(self):
        atlas, pixdims = self._atlases[self.gui.option("atlas").value]
        structure_maps = {}
        atlas_map = self._registry.loadAtlas(atlas.atlasID, loadSummary=False, resolution=pixdims[0])
        for idx, label in enumerate(atlas.labels):
            structure_maps[label.name] = fslimage_to_qpdata(atlas_map, vol=idx, name=label.name)
        return structure_maps

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
        self.gui.add("Number of voxels per patch (approx)", NumericOption(minval=1, maxval=1000, default=20, intonly=True), key="voxels-per-patch")

    @property
    def structures(self):
        return [
            Parameter("data", "Sequence of test values"),
        ]

    def get_simulated_data(self, data_model, param_values, output_param_maps=False):
        if len(param_values) != 1:
            raise QpException("Can only have a single structure in the checkerboard model")
        param_values = list(param_values.values())[0]

        param_values_list = {}
        varying_params = []
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

        repeats = [[0], [0], [0]]
        checkerboard_dims = []
        for idx, param in enumerate(varying_params):
            num_values = len(param_values_list[param])
            repeats[idx] = range(num_values)
            checkerboard_dims.append(patch_dims[idx] * num_values)
        for idx in range(len(varying_params), 3):
            checkerboard_dims.append(patch_dims[idx])

        output_data = None
        import itertools
        for indexes in itertools.product(*repeats):
            patch_values = dict(param_values)
            for idx, param in enumerate(varying_params):
                patch_values[param] = patch_values[param][indexes[idx]]
            
            timeseries = data_model.get_timeseries(patch_values)
            if output_data is None:
                output_data = np.zeros(list(checkerboard_dims) + [len(timeseries)], dtype=np.float32)
                if output_param_maps:
                    param_maps = {}
                    for param in param_values:
                        param_maps[param] = np.zeros(checkerboard_dims, dtype=np.float32)

            slices = []
            for dim_idx, patch_idx in enumerate(indexes):
                dim_length = patch_dims[dim_idx]
                slices.append(slice(patch_idx*dim_length, (patch_idx+1)*dim_length))
            output_data[slices] = timeseries
            if output_param_maps:
                for param, value in patch_values.items():
                    param_maps[param][slices] = value

        grid = DataGrid(checkerboard_dims, np.identity(4))
        sim_data = NumpyData(output_data, grid=grid, name="sim_data")

        if output_param_maps:
            for param in param_values: 
                param_maps[param] = NumpyData(param_maps[param], grid=grid, name=param)
            return sim_data, param_maps
        else:
            return sim_data
