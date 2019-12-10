"""
Perfusion simulation Quantiphyse plugin

Author: Martin Craig <martin.craig@eng.ox.ac.uk>
Copyright (c) 2016-2017 University of Oxford, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

try:
    from PySide import QtGui, QtCore, QtGui as QtWidgets
except ImportError:
    from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.gui.widgets import QpWidget, Citation, TitleWidget, RunWidget
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
        print("params map: structure=", self._structures)
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
                self._grid.addWidget(NumberListOption(initial=initial,
                                                      intonly=param.kwargs.get("intonly", False),
                                                      load_btn=False), param_idx + 1, structure_idx + 1)

    def _clear(self):
        while self._grid.count():
            child = self._grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class PerfSimWidget(QpWidget):
    """
    Perfusion simulation widget
    """
    def __init__(self, **kwargs):
        QpWidget.__init__(self, name="Perfusion Simulator", icon="perfsim", group="Simulation",
                          desc="Simulates data for various perfusion imaging sequences, from known parameter inputs", **kwargs)
        self._struc_models, self._data_models = {}, {}
        self._param_values = {}

        for name, cls in get_struc_models().items():
            self._struc_models[name] = cls(self.ivm)
        for name, cls in get_data_models().items():
            self._data_models[name] = cls(self.ivm)
    
    def init_ui(self):
        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        title = TitleWidget(self, help="perfsim", subtitle="Simulates data for various imaging sequences, from known parameter inputs %s" % __version__)
        main_vbox.addWidget(title)

        #cite = Citation(FAB_CITE_TITLE, FAB_CITE_AUTHOR, FAB_CITE_JOURNAL)
        #main_vbox.addWidget(cite)

        self._optbox = OptionBox()
        self._optbox.add("Structural model", ChoiceOption([m.display_name for m in self._struc_models.values()], self._struc_models.keys()), key="struc-model")
        self._optbox.add("Data model", ChoiceOption([m.display_name for m in self._data_models.values()], self._data_models.keys()), key="data-model")
        self._optbox.add("Additive noise (% of mean)", NumericOption(minval=0, maxval=200, default=10, intonly=True), checked=True, key="noise-percent")
        self._optbox.add("Also output clean data", TextOption("sim-data-clean"), checked=True, default=True, key="output-clean")
        self._optbox.option("struc-model").sig_changed.connect(self._struc_model_changed)
        self._optbox.option("data-model").sig_changed.connect(self._data_model_changed)
        self._optbox.option("noise-percent").sig_changed.connect(self._noise_changed)
        main_vbox.addWidget(self._optbox)

        # Create the GUIs for structural models - only one visible at a time!
        self.struc_model_guis = {}
        for model in self._struc_models.values():
            if model.gui is not None:
                hbox = QtGui.QHBoxLayout()
                struc_model_gui = QtGui.QGroupBox()
                struc_model_gui.setTitle(model.display_name)
                vbox = QtGui.QVBoxLayout()
                struc_model_gui.setLayout(vbox)
                vbox.addWidget(model.gui)
                hbox.addWidget(struc_model_gui)
                struc_model_gui.setVisible(False)
                main_vbox.addLayout(hbox)
                self.struc_model_guis[model.NAME] = struc_model_gui

        # Same for data models
        self.data_model_guis = {}
        for model in self._data_models.values():
            if model.gui is not None:
                hbox = QtGui.QHBoxLayout()
                data_model_gui = QtGui.QGroupBox()
                data_model_gui.setTitle(model.display_name)
                vbox = QtGui.QVBoxLayout()
                data_model_gui.setLayout(vbox)
                vbox.addWidget(model.gui)
                hbox.addWidget(data_model_gui)
                data_model_gui.setVisible(False)
                main_vbox.addLayout(hbox)
                self.data_model_guis[model.NAME] = data_model_gui

        # Box which will be used to enter parameter values
        self._params = ParamValuesGrid()
        main_vbox.addWidget(self._params)

        main_vbox.addWidget(RunWidget(self))
        main_vbox.addStretch(1)
        self._struc_model_changed()
        self._data_model_changed()
        self._noise_changed()

    def _struc_model_changed(self):
        struc_model = self._struc_models[self._optbox.option("struc-model").value]
        print("struc model", struc_model.NAME)
        for name, gui in self.struc_model_guis.items():
            print(name, struc_model.NAME)
            gui.setVisible(struc_model.NAME == name)
        self._params.structures = struc_model.structures
        
    def _data_model_changed(self):
        data_model = self._data_models[self._optbox.option("data-model").value]
        print("data model", data_model.NAME)
        for name, gui in self.data_model_guis.items():
            print(name, data_model.NAME)
            gui.setVisible(data_model.NAME == name)
        self._params.params = data_model.params

    def _noise_changed(self):
        self._optbox.set_visible("output-clean", self._optbox.option("noise-percent").isEnabled())

    def processes(self):
        opts = self._optbox.values()

        struc_model = self._struc_models[self._optbox.option("struc-model").value]
        data_model = self._data_models[self._optbox.option("data-model").value]
        opts["struc-model-options"] = struc_model.options
        opts["data-model-options"] = data_model.options
        opts["param-values"] = self._params.values
        print(opts)

        processes = [
            {"PerfSim" : opts}
        ]

        return processes