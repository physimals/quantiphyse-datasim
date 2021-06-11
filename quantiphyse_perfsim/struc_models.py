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

from quantiphyse.data import NumpyData, DataGrid, ImageVolumeManagement
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

    def resamp(self, qpdata):
        """
        Resample a map according to resampling options
        """
        resamp_options = dict(self.options.get("resampling", {}))
        if resamp_options:
            resamp_processes = get_plugins("processes", "ResampleProcess")
            if len(resamp_processes) != 1:
                raise QpException("Can't identify Resampling process")

            ivm = ImageVolumeManagement()
            ivm.add(qpdata)
            process = resamp_processes[0](ivm)
            resamp_options.update({
                "data" : qpdata.name,
                "output-name" : "output_res"
            })
            process.execute(resamp_options)
            while process.status == Process.RUNNING:
                time.sleep(1)

            if process.status == Process.FAILED:
                raise process.exception

            # FIXME hack
            process._complete()
            return ivm.data["output_res"]
        else:
            return qpdata

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

            timeseries = data_model.get_timeseries(single_values, pv_map.grid.shape)
            if output_data is None:
                output_data = np.zeros(list(pv_map.grid.shape) + [timeseries.shape[-1]], dtype=np.float32)
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

    @property
    def structures(self):
        ret = [struc for struc in self.default_strucs if struc.name in self.options["pvmaps"]]
        if self.options["additional"] is not None:
            for struc in self.options["additional"].values():
                ret.append(Parameter(**struc))
        return ret

    @property
    def structure_maps(self):
        pvmaps = self.options["pvmaps"]
        try:
            ret = {}
            total_pv = None
            grid = None
            for struc in self.default_strucs:
                if struc.name in pvmaps:
                    if grid is None:
                        # FIXME take grid from first data set - should there be a choice?
                        grid = self._ivm.data[pvmaps[struc.name]].grid
                    ret[struc.name] = self._ivm.data[pvmaps[struc.name]].resample(grid)
                    if total_pv is None:
                        total_pv = np.zeros(grid.shape, dtype=np.float32)
                    total_pv += ret[struc.name].raw()

            # Additional structures
            if self.options["additional"] is not None:
                for struc in self.options.get("additional", {}).values():
                    data = self._ivm.data[struc["pvmap"]].resample(grid)
                    struc_type = struc.get("struc_type", "")
                    if struc_type == "embed":
                        # Embedding - we need to downweight existing structure PVs so they all sum to 1 at most
                        pv = data.raw()
                        if "region" in struc:
                            # For an ROI we need to isolate the specific region and give it a PV of 1
                            pv[pv != struc["region"]] = 0
                            pv[pv > 0] = 1
                        pv = pv.astype(np.float32)
                        reweighting = 1-pv
                        for name, qpdata in ret.items():
                            qpdata = NumpyData(qpdata.raw() * reweighting, grid=qpdata.grid, name=name)
                            ret[name] = qpdata
                        ret[struc["name"]] = NumpyData(pv, grid=data.grid, name=struc["name"])
                    elif struc_type == "act":
                        # Activation mask - replace parent structure
                        parent_struc = struc.get("parent_struc", None)
                        if parent_struc is None:
                            raise QpException("Parent structure not defined for activation mask: %s" % struc["name"])
                        elif parent_struc not in ret:
                            raise QpException("Parent structure '%s' not found in structures list for activation mask: %s" % (parent_struc, struc["name"]))
                        parent_data = ret[parent_struc]
                        parent_data_masked = np.copy(parent_data.raw())
                        activation_mask = data.raw().astype(np.int)

                        # Activation structure takes over parent structure in the 
                        activation_data = np.zeros(parent_data_masked.shape, dtype=np.float32)
                        activation_data[activation_mask > 0] = parent_data_masked[activation_mask > 0]
                        parent_data_masked[activation_mask > 0] = 0
                        ret[parent_struc] = NumpyData(parent_data_masked, grid=parent_data.grid, name=parent_data.name)
                        ret[struc["name"]] = NumpyData(activation_data, grid=parent_data.grid, name=struc["name"])
                    elif struc_type == "add":
                        pass # Just use data directly
                        ret[struc["name"]] = data
                    else:
                        raise QpException("Unknown additional structure type: %s" % struc_type)

            # Resample PV maps according to specification
            # Note that resampling can lead to situations where the sum of the partial volumes
            # in the resampled maps is >1 in some voxels, even where this was not true in the
            # original maps. We need to detect this and rescale the affected voxels
            sum_map_pre = None
            sum_map_post = None
            for name in list(ret.keys()):
                pre = ret[name]
                if sum_map_pre is None:
                    sum_map_pre = np.zeros(pre.raw().shape, dtype=np.float32)
                sum_map_pre += pre.raw()

                post = self.resamp(pre)
                if sum_map_post is None:
                    sum_map_post = np.zeros(post.raw().shape, dtype=np.float32)
                sum_map_post += post.raw()
                ret[name] = post

            if sum_map_post is not None and np.all(sum_map_pre <= 1) and not np.all(sum_map_post <= 1):
                # Resampling has messed up the PV sum a bit - rescale to fix this but only in affected voxels
                self.debug("Max PV in resampled maps is %f: rescaling in %i voxels" % (np.max(sum_map_post), np.count_nonzero(sum_map_post > 1)))
                for name in list(ret.keys()):
                    pv_map = ret[name].raw()
                    pv_map[sum_map_post > 1] /= sum_map_post[sum_map_post > 1]

            return ret
        except KeyError as exc:
            raise
            #raise QpException("No structure map defined: %s" % str(exc.args[0]))

    def _reweight(self, strucs, to_pv):
        # FIXME
        pass

class FastStructureModel(PartialVolumeStructureModel):
    """
    Structural model which derives partial volume maps from a FAST segmentation
    """
    NAME = "fast"

    def __init__(self, ivm):
        StructureModel.__init__(self, ivm, "Partial volume maps from a FAST segmentation")
       
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
        if struc not in self._ivm.data:
            raise QpException("Structural image not loaded: %s" % struc)
        
        qpdata = self._ivm.data[struc]
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

    @property
    def structures(self):
        atlas, pixdims = self._atlases[self.options["atlas"]]
        structures = []
        for label in atlas.labels:
            structures.append(Parameter(label.name, label.name))
        return structures

    @property
    def structure_maps(self):
        atlas, pixdims = self._atlases[self.options["atlas"]]
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
