"""
Perfusion simulation Quantiphyse plugin

Perfusion simulation process

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

import numpy as np

from quantiphyse.processes import Process

from .data_models import get_data_model
from .struc_models import get_struc_model

class PerfSimProcess(Process):
    """
    Calculate CBF from R2P / DBV output
    """
    
    PROCESS_NAME = "PerfSim"
    
    def __init__(self, ivm, **kwargs):
        Process.__init__(self, ivm, **kwargs)

    def run(self, options):
        data_model = get_data_model(options["data-model"])
        struc_model = get_struc_model(options["struc-model"])

        struc_model_options = options.get("struc-model-options", {})
        pv_maps = struc_model.get_structure_maps(struc_model_options, self.ivm)
        print("pv_maps", pv_maps)

        param_values = options.get("param-values", {})
        print("param_values", param_values)

        data_model_options = options.get("data-model-options", {})
        data = data_model.generate_data(data_model_options, param_values, pv_maps)
        print("data", data)
        self.ivm.add(data, make_current=True)
