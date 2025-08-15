import sys
import platform
import os
import logging
import traceback


class LoggerService:
    """Service to write errors and warnings to an application log file."""

    logger = None

    def __init__(self):
        """
        Initialize the LoggerService, setting up file and console log handlers.

        Creates a log file in a platform-specific directory. If the directory does not exist, it is created.
        """
        app_path = None
        if platform.system() == 'Windows':
            home_path = os.path.expanduser("~")
            app_path = os.path.join(home_path, 'AppData', 'Roaming', 'ADIAT') + os.sep
            if not os.path.exists(app_path):
                os.makedirs(app_path, exist_ok=True)
        elif sys.platform == "darwin":
            home_path = os.path.expanduser("~")
            app_path = os.path.join(home_path, 'AppData', 'Roaming', 'ADIAT') + os.sep
            if not os.path.exists(app_path):
                os.makedirs(app_path, exist_ok=True)
        else:
            # Linux and others – use XDG config home or ~/.config/ADIAT
            home_path = os.path.expanduser("~")
            xdg = os.environ.get('XDG_CONFIG_HOME')
            base = xdg if xdg else os.path.join(home_path, '.config')
            app_path = os.path.join(base, 'ADIAT') + os.sep
            if not os.path.exists(app_path):
                os.makedirs(app_path, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        stdoutFmt = logging.Formatter(
            "%(name)s: %(asctime)s | %(levelname)s | %(process)d >>> %(message)s"
        )
        stdoutHandler.setFormatter(stdoutFmt)
        self.logger.addHandler(stdoutHandler)

        # File handler is optional if path is not writable
        try:
            log_path = os.path.join(app_path or '', 'adiat_logs.txt')
            fileHandler = logging.FileHandler(log_path)
            fileHandler.setFormatter(stdoutFmt)
            self.logger.addHandler(fileHandler)
        except Exception:
            pass

    def info(self, message):
        """
        Log a info message.

        Args:
            message (str): The info message to log.
        """
        self.logger.info(message)

    def warning(self, message):
        """
        Log a warning message.

        Args:
            message (str): The warning message to log.
        """
        self.logger.warning(message)

    def error(self, message):
        """
        Log an error message along with the traceback.

        Args:
            message (str): The error message to log.
        """
        self.logger.error(message)
