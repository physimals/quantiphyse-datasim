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
from .data_models import *
from .struc_models import *

#FAB_CITE_TITLE = "Variational Bayesian inference for a non-linear forward model"
#FAB_CITE_AUTHOR = "Chappell MA, Groves AR, Whitcher B, Woolrich MW."
#FAB_CITE_JOURNAL = "IEEE Transactions on Signal Processing 57(1):223-236, 2009."

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
        Mapping from param name to another mapping from structure name
        to parameter value (in that structure)
        """
        for structure_idx, structure in enumerate(self._structures):
            if structure.name not in self._values:
                self._values[structure.name] = {}
            for param_idx, param in enumerate(self._params):
                item = self._grid.itemAtPosition(param_idx + 1, structure_idx + 1)
                self._values[structure.name][param.name] = item.widget().value
        return self._values

    def _repopulate(self):
        self._clear()
        for structure_idx, structure in enumerate(self._structures):
            self._grid.addWidget(QtGui.QLabel(structure.display_name), 0, structure_idx + 1)

        for param_idx, param in enumerate(self._params):
            self._grid.addWidget(QtGui.QLabel(param.display_name), param_idx + 1, 0)
            for structure_idx, structure in enumerate(self._structures):
                # FIXME use existing values / parameter defaults
                self._grid.addWidget(NumericOption(slider=False, **param.kwargs), param_idx + 1, structure_idx + 1)

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
        self._struc_models = [
            Fsl1mmModel(self.ivm),
            UserPvModel(self.ivm),
        ]
        self._data_models = [
            AslDataModel(self.ivm),
        ]
        self._param_values = {
        }

    def init_ui(self):
        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        title = TitleWidget(self, help="perfsim", subtitle="Simulates data for various imaging sequences, from known parameter inputs %s" % __version__)
        main_vbox.addWidget(title)

        #cite = Citation(FAB_CITE_TITLE, FAB_CITE_AUTHOR, FAB_CITE_JOURNAL)
        #main_vbox.addWidget(cite)

        self._optbox = OptionBox()
        self._optbox.add("Structural model", ChoiceOption([m.display_name for m in self._struc_models], [m.name for m in self._struc_models]), key="struc-model")
        self._optbox.add("Data model", ChoiceOption([m.display_name for m in self._data_models], [m.name for m in self._data_models]), key="data-model")
        self._optbox.option("struc-model").sig_changed.connect(self._struc_model_changed)
        self._optbox.option("data-model").sig_changed.connect(self._data_model_changed)
        main_vbox.addWidget(self._optbox)

        # Create the GUIs for structural models - only one visible at a time!
        self.struc_model_guis = {}
        for model in self._struc_models:
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
                self.struc_model_guis[model.name] = struc_model_gui

        # Same for data models
        self.data_model_guis = {}
        for model in self._data_models:
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
                self.data_model_guis[model.name] = data_model_gui

        # Box which will be used to enter parameter values
        self._params = ParamValuesGrid()
        main_vbox.addWidget(self._params)

        main_vbox.addWidget(RunWidget(self))
        main_vbox.addStretch(1)
        self._struc_model_changed()
        self._data_model_changed()

    def _struc_model_changed(self):
        struc_model = self._get_model(self._struc_models, self._optbox.option("struc-model").value)
        print("struc model", struc_model.name)
        for name, gui in self.struc_model_guis.items():
            print(name, struc_model.name)
            gui.setVisible(struc_model.name == name)
        self._params.structures = struc_model.structures
        
    def _data_model_changed(self):
        data_model = self._get_model(self._data_models, self._optbox.option("data-model").value)
        print("data model", data_model.name)
        for name, gui in self.data_model_guis.items():
            print(name, data_model.name)
            gui.setVisible(data_model.name == name)
        self._params.params = data_model.params

    def _get_model(self, models, name):
        for model in models:
            if model.name == name:
                return model
        return None

    def processes(self):
        opts = self._optbox.values()

        struc_model = self._get_model(self._struc_models, self._optbox.option("struc-model").value)
        data_model = self._get_model(self._data_models, self._optbox.option("data-model").value)
        print(struc_model)
        print(data_model)
        opts["struc-model-options"] = struc_model.options
        opts["data-model-options"] = data_model.options
        opts["param-values"] = self._params.values

        print(opts)

        processes = [
            {"PerfSim" : opts}
        ]

        return processes