import os
import shutil

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from core.views.components.Preferences_ui import Ui_Preferences
from core.services.SettingsService import SettingsService
from helpers.PickleHelper import PickleHelper
from core.services.AOICoordinateService import AOICoordinateService, AOICoordinateError


class Preferences(QDialog, Ui_Preferences):
    """Controller for the Preferences dialog box.

    This class manages user preferences for the application, including settings such as
    maximum areas of interest, theme, area of interest radius, position format, and temperature unit.
    """

    def __init__(self, parent):
        """Initializes the Preferences dialog.

        Args:
            parent (MainWindow): The parent window for the dialog.
        """
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self._load_settings()
        self._connect_signals()

    def _load_settings(self):
        """Loads the settings from SettingsService and updates the UI accordingly."""
        self.maxAOIsSpinBox.setValue(self.parent.settings_service.get_setting('MaxAOIs'))
        self.themeComboBox.setCurrentText(self.parent.settings_service.get_setting('Theme'))
        self.AOIRadiusSpinBox.setValue(self.parent.settings_service.get_setting('AOIRadius'))
        self.positionFormatComboBox.setCurrentText(self.parent.settings_service.get_setting('PositionFormat'))
        self.temperatureComboBox.setCurrentText(self.parent.settings_service.get_setting('TemperatureUnit'))
        self.distanceComboBox.setCurrentText(self.parent.settings_service.get_setting('DistanceUnit'))
        self.demPathLineEdit.setText(self.parent.settings_service.get_setting('DEMPath') or "")
        drone_sensor_version = PickleHelper.get_drone_sensor_file_version()
        self.dronSensorVersionLabel.setText(f"{drone_sensor_version['Version']}_{drone_sensor_version['Date']}")

    def _connect_signals(self):
        """Connects UI signals to the appropriate update methods."""
        self.maxAOIsSpinBox.valueChanged.connect(self._update_max_aois)
        self.AOIRadiusSpinBox.valueChanged.connect(self._update_aoi_radius)
        self.themeComboBox.currentTextChanged.connect(self._update_theme)
        self.positionFormatComboBox.currentTextChanged.connect(self._update_position_format)
        self.temperatureComboBox.currentTextChanged.connect(self._update_temperature_unit)
        self.distanceComboBox.currentTextChanged.connect(self._update_distance_unit)
        self.demPathButton.clicked.connect(self._dem_path_button_clicked)
        self.demPathLineEdit.editingFinished.connect(self._update_dem_path)
        self.droneSensorButton.clicked.connect(self._droneSensorButton_clicked)

    def _update_max_aois(self):
        """Updates the maximum areas of interest setting based on the spinbox value."""
        self.parent.settings_service.set_setting('MaxAOIs', self.maxAOIsSpinBox.value())

    def _update_aoi_radius(self):
        """Updates the area of interest radius setting based on the spinbox value."""
        self.parent.settings_service.set_setting('AOIRadius', self.AOIRadiusSpinBox.value())

    def _update_theme(self):
        """Updates the theme setting based on the selected combobox value and applies it."""
        theme = self.themeComboBox.currentText()
        self.parent.settings_service.set_setting('Theme', theme)
        self.parent.update_theme(theme)

    def _update_position_format(self):
        """Updates the position format setting based on the selected combobox value."""
        self.parent.settings_service.set_setting('PositionFormat', self.positionFormatComboBox.currentText())

    def _update_temperature_unit(self):
        """Updates the temperature unit setting based on the selected combobox value."""
        self.parent.settings_service.set_setting('TemperatureUnit', self.temperatureComboBox.currentText())

    def _update_distance_unit(self):
        """Updates the distance unit setting based on the selected combobox value."""
        self.parent.settings_service.set_setting('DistanceUnit', self.distanceComboBox.currentText())

    def _update_dem_path(self):
        """Updates the DEM directory path based on the line edit value."""
        path = self.demPathLineEdit.text()
        if self._validate_dem_path(path):
            self.parent.settings_service.set_setting('DEMPath', path)
        else:
            self.demPathLineEdit.setText("")
            self.parent.settings_service.set_setting('DEMPath', "")

    def _dem_path_button_clicked(self):
        """Opens a directory dialog for selecting the DEM directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select DEM Directory", self.demPathLineEdit.text())
        if directory:
            if self._validate_dem_path(directory):
                self.demPathLineEdit.setText(directory)
                self.parent.settings_service.set_setting('DEMPath', directory)

    def _validate_dem_path(self, path: str) -> bool:
        """Validate the DEM path ensuring at least one usable DEM file exists."""
        if not path:
            return True
        try:
            AOICoordinateService(path)
            return True
        except (AOICoordinateError, Exception):
            QMessageBox.critical(self, "Invalid DEM Path", "The selected path does not contain valid DEM files.")
            return False

    def _droneSensorButton_clicked(self):
        """
        Opens a file dialog for the user to select a .pkl file and copies it to the app's destination directory.
        """
        # Only allow .pkl files
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a Drone Sensor File",
            "",
            "Pickle Files (*.pkl)"
        )
        if not filename:
            return  # User cancelled

        # Destination path (change as needed)
        dest_dir = PickleHelper._get_destination_path()
        dest_file = os.path.join(dest_dir, 'drones.pkl')
        shutil.copy(filename, dest_file)
        PickleHelper.force_reload()
        drone_sensor_version = PickleHelper.get_drone_sensor_file_version()
        self.dronSensorVersionLabel.setText(f"{drone_sensor_version['Version']}_{drone_sensor_version['Date']}")
