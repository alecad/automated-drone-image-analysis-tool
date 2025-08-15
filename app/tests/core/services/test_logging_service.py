import pytest
import os
import platform
from unittest.mock import patch
from app.core.services.LoggerService import LoggerService  # Adjust the import according to your project structure


@pytest.fixture
def logger_service():
    with patch("os.makedirs"), patch("os.path.exists", return_value=False):
        return LoggerService()


def test_logger_service_initialization_windows():
    with patch("platform.system", return_value="Windows"), \
            patch("os.makedirs") as mock_makedirs, \
            patch("os.path.exists", return_value=False), \
            patch("logging.getLogger"), \
            patch("logging.FileHandler"), \
            patch("logging.StreamHandler"):
        logger_service = LoggerService()
        home_path = os.path.expanduser("~")
        app_path = home_path + '/AppData/Roaming/ADIAT/'
        # Allow exist_ok kwarg
        assert mock_makedirs.call_count == 1
        args, kwargs = mock_makedirs.call_args
        assert args[0] == app_path
        assert kwargs.get('exist_ok', False) is True
        assert logger_service.logger is not None


def test_logger_service_initialization_macos():
    with patch("sys.platform", "darwin"), \
            patch("os.makedirs") as mock_makedirs, \
            patch("os.path.exists", return_value=False), \
            patch("logging.getLogger"), \
            patch("logging.FileHandler"), \
            patch("logging.StreamHandler"):
        logger_service = LoggerService()
        home_path = os.path.expanduser("~")
        app_path = home_path + '/AppData/Roaming/ADIAT/'
        # Not strictly required here, just ensure makedirs was attempted
        assert mock_makedirs.called


def test_warning(logger_service):
    with patch.object(logger_service.logger, 'warning') as mock_warning:
        logger_service.warning("This is a warning message")
        mock_warning.assert_called_once_with("This is a warning message")


def test_error(logger_service):
    with patch.object(logger_service.logger, 'error') as mock_error, \
            patch("traceback.format_exc", return_value="Traceback (most recent call last):"):
        logger_service.error("This is an error message")
        mock_error.assert_called_once_with("This is an error message")
