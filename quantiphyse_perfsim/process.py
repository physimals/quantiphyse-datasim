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

        data_model_options = options.get("data-model-options", {})
        print(data_model_options)
        data_model.options = data_model_options

        struc_model_name = options.pop("struc-model", None)
        struc_model_options = options.get("struc-model-options", {})
        if struc_model_name is None:
            raise QpException("Structure model not specified")

        struc_model = self._struc_models.get(struc_model_name, None)
        if struc_model is None:
            raise QpException("Unknown structure model: %s" % struc_model_name)
        struc_model = struc_model(self.ivm)
        struc_model.options = struc_model_options

        param_values = options.get("param-values", {})
        print("param_values", param_values)
        clean_data = struc_model.get_simulated_data(data_model, param_values)

        noise = options.pop("noise-percent", 0)
        if noise > 0:
            noise_std = np.mean(clean_data.raw()) * float(noise) / 100 
            random_noise = np.random.normal(0, noise_std, clean_data.raw().shape)
            noisy_data = NumpyData(clean_data.raw() + random_noise, grid=clean_data.grid, name="sim_data")
            self.ivm.add(noisy_data, make_current=True)
            if options.pop("output-clean", False):
                self.ivm.add(clean_data, name="sim_data_clean", make_current=False)
        else:
            self.ivm.add(clean_data, name="sim_data", make_current=True)
