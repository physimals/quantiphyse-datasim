"""
Perfusion simulation Quantiphyse plugin

Perfusion simulation process

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

import numpy as np

from quantiphyse.processes import Process
from quantiphyse.utils import QpException
from quantiphyse.data import NumpyData

from .data_models import get_data_models
from .struc_models import get_struc_models

class PerfSimProcess(Process):
    """
    Calculate CBF from R2P / DBV output
    """
    
    PROCESS_NAME = "PerfSim"
    
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
        self.log("Created data model: %s\n" % data_model_name)

        data_model_options = options.get("data-model-options", {})
        data_model.options = data_model_options

        struc_model_name = options.pop("struc-model", None)
        struc_model_options = options.get("struc-model-options", {})
        if struc_model_name is None:
            raise QpException("Structure model not specified")
        self.log("Created structure model: %s\n" % struc_model_name)

        struc_model = self._struc_models.get(struc_model_name, None)
        if struc_model is None:
            raise QpException("Unknown structure model: %s" % struc_model_name)
        struc_model = struc_model(self.ivm)
        struc_model.options = struc_model_options

        param_values = options.get("param-values", {})
        output_param_maps = options.pop("output-param-maps", False)
        self.log("Getting simulated data\n")
        ret = struc_model.get_simulated_data(data_model, param_values, output_param_maps=output_param_maps)
        if output_param_maps:
            clean_data, param_maps = ret
        else:
            clean_data, param_maps = ret, {}

        output_name = options.pop("output", "sim_data")
        noise = options.pop("noise-percent", 0)
        if noise > 0:
            self.log("Adding noise\n")
            noise_std = np.mean(clean_data.raw()) * float(noise) / 100 
            random_noise = np.random.normal(0, noise_std, clean_data.raw().shape)
            noisy_data = NumpyData(clean_data.raw() + random_noise, grid=clean_data.grid, name=output_name)
            self.ivm.add(noisy_data, make_main=True)
            output_clean = options.pop("output-clean", None)
            if output_clean:
                self.ivm.add(clean_data, name=output_clean)
        else:
            self.ivm.add(clean_data, name=output_name, make_main=True)

        for param, qpdata in param_maps.items():
            self.ivm.add(qpdata, name=param)
