"""
Data simulation Quantiphyse plugin

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.gui.widgets import QpWidget, Citation, TitleWidget, RunWidget, WarningBox
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, ChoiceOption
from quantiphyse.utils import QpException

from ._version import __version__
from .data_models import get_data_models
from .struc_models import get_struc_models
from . import data_model_views
from . import struc_model_views

def get_view_class(model_class):
    view_name = model_class.__name__ + "View"
    cls = getattr(data_model_views, view_name, None)
    if cls is None:
        cls = getattr(struc_model_views, view_name, None)
    return cls

class ParamValuesGrid(QtGui.QGroupBox):
    """
    Widget which presents a grid of values so the user can specify the value of 
    each parameter in each partial volume structure
    """

    def __init__(self, title="Parameter Values"):
        QtGui.QGroupBox.__init__(self, title)
        self._grid = QtGui.QGridLayout()
        self.setLayout(self._grid)
        self._structures = []
        self._params = []
        self._values = {}
        
    @property
    def structures(self):
        """
        A sequence of named structures
        """
        return self._structures

    @structures.setter
    def structures(self, structures):
        structures = sorted(structures, key=lambda x: x.name)
        if structures != self._structures:
            self._structures = structures
            self._repopulate()

    @property
    def params(self):
        """
        A sequence of named parameters
        """
        return self._params

    @params.setter
    def params(self, params):
        if params != self._params:
            self._params = params
            self._repopulate()

    @property
    def values(self):
        """
        Mapping from structure name to another mapping from param name
        to parameter value(s) in that structure
        """
        ret = {}
        for structure_idx, structure in enumerate(self._structures):
            ret[structure.name] = {}
            if structure.name not in self._values:
                self._values[structure.name] = {}
            for param_idx, param in enumerate(self._params):
                item = self._grid.itemAtPosition(param_idx + 1, structure_idx + 1)
                vals = item.widget().value
                if len(vals) == 1:
                    vals = vals[0]
                self._values[structure.name][param.name] = vals

                # We keep records in self._values for data structures/params that aren't in the
                # current list, so need to be careful to only return the ones that are
                ret[structure.name][param.name] = vals

        return ret

    def _repopulate(self):
        self._clear()
        for structure_idx, structure in enumerate(self._structures):
            self._grid.addWidget(QtGui.QLabel(structure.display_name), 0, structure_idx + 1)

        for param_idx, param in enumerate(self._params):
            self._grid.addWidget(QtGui.QLabel(param.display_name), param_idx + 1, 0)
            for structure_idx, structure in enumerate(self._structures):
                # The initial value for a parameter in a structure is the previous value set (if any), 
                # the default value for that structure (if any), the default value for the parent
                # structure (if any), the generic parameter default (if any), and if none of the above, zero.
                if structure.name in self._values and param.name in self._values[structure.name]:
                    initial = self._values[structure.name][param.name]
                elif structure.name in param.kwargs.get("struc_defaults", {}):
                    initial = param.kwargs["struc_defaults"][structure.name]
                elif structure.kwargs.get("parent_struc", "") in param.kwargs.get("struc_defaults", {}):
                    initial = param.kwargs["struc_defaults"][structure.kwargs.get("parent_struc", "")]
                else:
                    initial = [param.kwargs.get("default", 0.0)]
                if isinstance(initial, (int, float)):
                    initial = [initial]
                self._grid.addWidget(NumberListOption(initial=initial,
                                                      intonly=param.kwargs.get("intonly", False),
                                                      load_btn=False), param_idx + 1, structure_idx + 1)

    def _clear(self):
        while self._grid.count():
            child = self._grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class OptionsWidget(QtGui.QWidget):

    sig_changed = QtCore.Signal()

    def __init__(self, ivm, parent):
        QtGui.QWidget.__init__(self, parent)
        self.ivm = ivm

class ModelOptions(OptionsWidget):
    def __init__(self, ivm, parent, model_type, abbrev, model_classes):
        OptionsWidget.__init__(self, ivm, parent)
        self._views = {}
        self.model = None
        self.view = None
        self._option_name = "%s-model" % abbrev
        for name, cls in model_classes.items():
            self._views[name] = get_view_class(cls)(ivm)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("%s model" % model_type, ChoiceOption([v.model.display_name for v in self._views.values()], self._views.keys()), key=self._option_name)
        self.options.option(self._option_name).sig_changed.connect(self._model_changed)
        main_vbox.addWidget(self.options)

        self._create_guis(main_vbox)
        main_vbox.addStretch(1)
        self._model_changed()

    def _create_guis(self, main_vbox):
        # Create the GUIs for models - only one visible at a time!
        for view in self._views.values():
            if view.gui is not None:
                view.gui.setVisible(False)
                if isinstance(view.gui, QtGui.QWidget):
                    main_vbox.addWidget(view.gui)
                else:
                    main_vbox.addLayout(view.gui)
                view.model.options.sig_changed.connect(self._model_option_changed)
                
    def _model_changed(self):
        chosen_name = self.options.option(self._option_name).value
        self.view = self._views[chosen_name]
        self.model = self.view.model
        for name, view in self._views.items():
            view.gui.setVisible(chosen_name == name)
        self.sig_changed.emit()

    def _model_option_changed(self, _key, _value):
        self.sig_changed.emit()

class NoiseOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("Add noise with SNR", NumericOption(minval=0.1, maxval=100, default=10), checked=True, key="snr")
        self.options.sig_changed.connect(self.sig_changed.emit)
        main_vbox.addWidget(self.options)

        main_vbox.addStretch(1)

class MotionOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("Simulate motion", BoolOption(default=False), key="motion")
        self.options.option("motion").sig_changed.connect(self._update_widget_visibility)
        self.options.add("Random translation standard deviation (mm)", NumericOption(minval=0, maxval=5, default=1, decimals=2), key="std")
        self.options.add("Random rotation standard deviation (\N{DEGREE SIGN})", NumericOption(minval=0, maxval=10, default=1, decimals=2), key="std_rot")
        self.options.add("Padding (mm)", NumericOption(minval=0, maxval=10, default=5, decimals=1), key="padding", checked=True)
        self.options.add("Interpolation", ChoiceOption(["Nearest neighbour", "Linear", "Quadratic", "Cubic"], return_values=range(4), default=3), key="order")
        main_vbox.addWidget(self.options)

        main_vbox.addStretch(1)
        self._update_widget_visibility()

    def _update_widget_visibility(self):
        enabled = self.options.option("motion").value
        for option in ["std", "std_rot", "padding", "order"]:
            self.options.set_visible(option, enabled)

class ParamsOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self._params = ParamValuesGrid()
        main_vbox.addWidget(self._params)
        main_vbox.addStretch(1)

    def set_models(self, data_model, struc_model):
        self._params.params = data_model.params
        self._params.structures = struc_model.structures

    @property
    def options(self):
        return self._params.values

class OutputOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("Output name", TextOption("sim_data"), key="output")
        self.options.add("Output parameter maps", BoolOption(), default=False, key="output-param-maps")
        self.options.add("Output clean data (no noise/motion)", TextOption("sim_data_clean"), checked=True, default=True, key="output-clean")
        main_vbox.addWidget(self.options)
        main_vbox.addStretch(1)

class DataSimWidget(QpWidget):
    """
    Data simulation widget
    """
    def __init__(self, **kwargs):
        QpWidget.__init__(self, name="Data Simulator", icon="datasim", group="Simulation",
                          desc="Simulates data for various imaging sequences, from known parameter inputs", **kwargs)
        self._param_values = {}

    def init_ui(self):
        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        title = TitleWidget(self, help="generic/datasim", subtitle="Simulates data for various imaging sequences, from known parameter inputs %s" % __version__)
        main_vbox.addWidget(title)

        self.tabs = QtGui.QTabWidget()
        main_vbox.addWidget(self.tabs)

        self.struc_model = ModelOptions(self.ivm, parent=self, model_type="Structural", abbrev="struc", model_classes=get_struc_models())
        self.struc_model.sig_changed.connect(self._update_params)
        self.data_model = ModelOptions(self.ivm, parent=self, model_type="Data", abbrev="data", model_classes=get_data_models())
        self.data_model.sig_changed.connect(self._update_params)
        self.params = ParamsOptions(self.ivm, parent=self)
        self.noise = NoiseOptions(self.ivm, parent=self)
        self.motion = MotionOptions(self.ivm, parent=self)
        self.output = OutputOptions(self.ivm, parent=self)
        self.tabs.addTab(self.struc_model, "Structure model")
        self.tabs.addTab(self.data_model, "Data model")
        self.tabs.addTab(self.params, "Parameter values")
        self.tabs.addTab(self.noise, "Noise")
        self.tabs.addTab(self.motion, "Motion")
        self.tabs.addTab(self.output, "Output")

        main_vbox.addStretch(1)
        main_vbox.addWidget(RunWidget(self))
        self._update_params()

    def _update_params(self):
        self.params.set_models(self.data_model.model, self.struc_model.model)

    def processes(self):
        processes = []
        opts = self.output.options.values()
        opts.update(self.struc_model.options.values())
        opts.update(self.data_model.options.values())
        opts["struc-model-options"] = self.struc_model.model.options
        opts["data-model-options"] = self.data_model.model.options
        opts["param-values"] = self.params.options
        processes.append({"DataSim" : opts})

        motion_opts = self.motion.options.values()
        if motion_opts.pop("motion"):
            motion_opts.update({
                "data" : opts["output"],
                "output-name" : opts["output"],
            })
            processes.append({"SimMotion" : motion_opts})

        noise_opts = self.noise.options.values()
        if "snr" in noise_opts:
            noise_opts.update({
                "data" : opts["output"],
                "roi" : opts["output"] + "_roi",
                "output-name" : opts["output"],
            })
            # Hack to support ASL TC data where the 'signal' is the difference
            # not the magnitude of the static signal
            if opts["data-model"] == "asl" and opts["data-model-options"].get("iaf", "diff") in ("tc", "ct"):
                noise_opts["mode"] = "diff"
            processes.append({"AddNoise" : noise_opts})

        processes.append({"Delete" : {opts["output"] + "_roi" : ""}})

        return processes