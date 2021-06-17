"""
Data simulation Quantiphyse plugin

Data simulation process

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

import numpy as np

from quantiphyse.data import NumpyData
from quantiphyse.processes import Process
from quantiphyse.utils import QpException

from .data_models import get_data_models
from .struc_models import get_struc_models

class DataSimProcess(Process):
    """
    Calculate CBF from R2P / DBV output
    """
    
    PROCESS_NAME = "DataSim"
    
    def __init__(self, ivm, **kwargs):
        Process.__init__(self, ivm, **kwargs)
        self._struc_models = get_struc_models()
        self._data_models = get_data_models()

    def run(self, options):
        data_model_name = options.pop("data-model", None)
        if data_model_name is None:
            raise QpException("Data model not specified")
        data_model = self._data_models.get(data_model_name, None)
        if data_model is None:
            raise QpException("Unknown data model: %s" % data_model_name)
        data_model = data_model(self.ivm)
        data_model_options = options.pop("data-model-options", {})
        data_model.options = data_model_options
        self.log("Created data model: %s\n" % data_model_name)

        struc_model_name = options.pop("struc-model", None)
        struc_model_options = options.pop("struc-model-options", {})
        if struc_model_name is None:
            raise QpException("Structure model not specified")
        struc_model = self._struc_models.get(struc_model_name, None)
        if struc_model is None:
            raise QpException("Unknown structure model: %s" % struc_model_name)
        struc_model = struc_model(self.ivm)
        struc_model.options = struc_model_options
        self.log("Created structure model: %s\n" % struc_model_name)

        param_values = options.pop("param-values", {})
        output_param_maps = options.pop("output-param-maps", False)
        self.log("Getting simulated data\n")
        ret = struc_model.get_simulated_data(data_model, param_values, output_param_maps=output_param_maps)
        if output_param_maps:
            sim_data, param_maps = ret
        else:
            sim_data, param_maps = ret, {}

        output_clean_name = options.pop("output-clean", "")
        if output_clean_name:
            self.ivm.add(sim_data.raw().copy(), grid=sim_data.grid, name=output_clean_name, make_current=False)

        for param, qpdata in param_maps.items():
            self.ivm.add(qpdata, name=param, make_current=False)

        output_name = options.pop("output", "sim_data")
        self.ivm.add(sim_data, name=output_name, make_current=True)

        output_roi = output_name + "_roi"
        if sim_data.ndim > 3:
            roi_data = sim_data.raw()[..., 0]
        else:
            roi_data = sim_data.raw()
        roi = NumpyData(np.array(roi_data > 0, dtype=np.int), grid=sim_data.grid, roi=True, name=output_roi)
        self.ivm.add(roi, make_current=False)
