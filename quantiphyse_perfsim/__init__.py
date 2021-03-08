"""
Quantiphyse plugin for perfusion simulation

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""
from .widget import PerfSimWidget
from .process import PerfSimProcess
from .data_models import *
from .data_model_views import *
from .struc_models import *
from .struc_model_views import *

def get_view_class(model_class):
    view_name = model_class.__name__ + "View"
    return globals().get(view_name, None)

QP_MANIFEST = {
    "widgets" : [
        PerfSimWidget
    ],
    "processes" : [
        PerfSimProcess
    ],
    "perfsim-data-models" : [
        SpinEchoDataModel,
        AslDataModel,
        DscDataModel,
        DceDataModel,
    ],
    "perfsim-struc-models" : [
        UserPvModel,
        CheckerboardModel,
        #FslStdStructureModel,
        FastStructureModel,
    ]
}
