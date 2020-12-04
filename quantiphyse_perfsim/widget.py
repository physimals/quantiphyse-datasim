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

from ._version import __version__
from .data_models import get_data_models
from .struc_models import get_struc_models

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
            if structure.name not in self._values:
                self._values[structure.name] = {}
            for param_idx, param in enumerate(self._params):
                item = self._grid.itemAtPosition(param_idx + 1, structure_idx + 1)
                vals = item.widget().value
                if len(vals) == 1:
                    vals = vals[0]
                self._values[structure.name][param.name] = vals

            # We keep records of test values for data structures that aren't in the
            # current list, so need to be careful to only return the ones that are
            ret[structure.name] = dict(self._values[structure.name])

        return ret

    def _repopulate(self):
        self._clear()
        for structure_idx, structure in enumerate(self._structures):
            self._grid.addWidget(QtGui.QLabel(structure.display_name), 0, structure_idx + 1)

        for param_idx, param in enumerate(self._params):
            self._grid.addWidget(QtGui.QLabel(param.display_name), param_idx + 1, 0)
            for structure_idx, structure in enumerate(self._structures):
                if structure.name in self._values and param.name in self._values[structure.name]:
                    initial = self._values[structure.name][param.name]
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
        self._models = {}
        self.model = None
        self._option_name = "%s-model" % abbrev
        for name, cls in model_classes.items():
            self._models[name] = cls(self.ivm)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("%s model" % model_type, ChoiceOption([m.display_name for m in self._models.values()], self._models.keys()), key=self._option_name)
        self.options.option(self._option_name).sig_changed.connect(self._changed)
        main_vbox.addWidget(self.options)

        self._create_guis(main_vbox)
        main_vbox.addStretch(1)
        self._changed()

    def _create_guis(self, main_vbox):
        # Create the GUIs for models - only one visible at a time!
        for model in self._models.values():
            if model.gui is not None:
                model.gui.setVisible(False)
                if isinstance(model.gui, QtGui.QWidget):
                    main_vbox.addWidget(model.gui)
                else:
                    main_vbox.addLayout(model.gui)

    def _changed(self):
        chosen_name = self.options.option(self._option_name).value
        self.model = self._models[chosen_name]
        for name, model in self._models.items():
            model.gui.setVisible(chosen_name == name)
        self.sig_changed.emit()

class NoiseOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        self.options = OptionBox()
        self.options.add("Additive noise (% of mean)", NumericOption(minval=0, maxval=200, default=10, intonly=True), checked=True, key="noise-percent")
        self.options.option("noise-percent").sig_changed.connect(self._changed)
        main_vbox.addWidget(self.options)

        main_vbox.addStretch(1)
        self._changed()

    def _changed(self):
        pass
        #self.options.set_visible("output-clean", self.options.option("noise-percent").isEnabled())
        self.sig_changed.emit()

class MotionOptions(OptionsWidget):
    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

class PerfSimWidget(QpWidget):
    """
    Data simulation widget
    """
    def __init__(self, **kwargs):
        QpWidget.__init__(self, name="Data Simulator", icon="perfsim", group="Simulation",
                          desc="Simulates data for various imaging sequences, from known parameter inputs", **kwargs)
        self._param_values = {}

    def init_ui(self):
        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        title = TitleWidget(self, help="generic/perfsim", subtitle="Simulates data for various imaging sequences, from known parameter inputs %s" % __version__)
        main_vbox.addWidget(title)

        self.tabs = QtGui.QTabWidget()
        main_vbox.addWidget(self.tabs)

        self.struc_model = ModelOptions(self.ivm, parent=self, model_type="Structural", abbrev="struc", model_classes=get_struc_models())
        self.struc_model.sig_changed.connect(self._changed)
        self.tabs.addTab(self.struc_model, "Structure model")
        self.data_model = ModelOptions(self.ivm, parent=self, model_type="Data", abbrev="data", model_classes=get_data_models())
        self.data_model.sig_changed.connect(self._changed)
        self.tabs.addTab(self.data_model, "Data model")
        self.noise = NoiseOptions(self.ivm, parent=self)
        self.noise.sig_changed.connect(self._changed)
        self.tabs.addTab(self.noise, "Noise")
        self.motion = MotionOptions(self.ivm, parent=self)
        self.tabs.addTab(self.motion, "Motion")

        self.options = OptionBox()
        self.options.add("Output name", TextOption("sim_data"), key="output")
        self.options.add("Also output clean data", TextOption("sim_data_clean"), checked=True, default=True, key="output-clean")
        self.options.add("Output parameter maps", BoolOption(), default=False, key="output-param-maps")
        main_vbox.addWidget(self.options)

        # Box which will be used to enter parameter values
        self._params = ParamValuesGrid()
        main_vbox.addWidget(self._params)

        main_vbox.addWidget(RunWidget(self))
        main_vbox.addStretch(1)

    def _changed(self):
        self._params.params = self.data_model.model.params
        self._params.structures = self.struc_model.model.structures

    def processes(self):
        opts = self.options.values()
        opts.update(self.struc_model.options.values())
        opts.update(self.data_model.options.values())
        opts.update(self.noise.options.values())
        opts["struc-model-options"] = self.struc_model.model.options
        opts["data-model-options"] = self.data_model.model.options
        opts["param-values"] = self._params.values

        processes = [
            {"PerfSim" : opts}
        ]

        return processes