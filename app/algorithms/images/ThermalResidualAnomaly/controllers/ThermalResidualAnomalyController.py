from PySide6.QtWidgets import QWidget

from algorithms.AlgorithmController import AlgorithmController
from algorithms.images.ThermalResidualAnomaly.views.ThermalResidualAnomaly_ui import Ui_ThermalResidualAnomaly
from core.services.SettingsService import SettingsService


class ThermalResidualAnomalyController(QWidget, Ui_ThermalResidualAnomaly, AlgorithmController):
    """Controller for the Thermal Residual Anomaly algorithm widget."""

    def __init__(self, config, theme):
        QWidget.__init__(self)
        AlgorithmController.__init__(self, config)
        self.settings_service = SettingsService()
        self.setupUi(self)
        self.sensitivitySlider.valueChanged.connect(self.update_sensitivity)

    def get_options(self):
        options = dict()
        options['sensitivity'] = int(self.sensitivityValueLabel.text())
        options['type'] = self.anomalyTypeComboBox.currentText()
        return options

    def update_sensitivity(self):
        self.sensitivityValueLabel.setText(str(self.sensitivitySlider.value()))

    def _threshold_to_sensitivity(self, threshold):
        threshold = float(threshold)
        mapping = {
            1: 8.0,
            2: 7.0,
            3: 6.0,
            4: 5.0,
            5: 4.0,
            6: 3.0,
            7: 2.0,
            8: 1.7,
            9: 1.3,
            10: 1.0,
        }
        return min(mapping, key=lambda value: abs(mapping[value] - threshold))

    def validate(self):
        return None

    def load_options(self, options):
        if not isinstance(options, dict):
            return

        if 'sensitivity' in options:
            sensitivity = max(1, min(10, int(options['sensitivity'])))
            self.sensitivitySlider.setProperty("value", sensitivity)
            self.sensitivityValueLabel.setText(str(sensitivity))
        if 'threshold' in options:
            sensitivity = self._threshold_to_sensitivity(options['threshold'])
            self.sensitivitySlider.setProperty("value", sensitivity)
            self.sensitivityValueLabel.setText(str(sensitivity))
        if 'type' in options:
            self.anomalyTypeComboBox.setCurrentText(options['type'])

        # Legacy no-op: tolerate historic settings that still include segments.
        _ = options.get('segments')
