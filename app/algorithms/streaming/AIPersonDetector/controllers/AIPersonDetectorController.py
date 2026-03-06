"""Streaming AI Person Detector controller."""

from typing import Any, Dict, List
import numpy as np

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QVBoxLayout

from core.controllers.streaming.base import StreamAlgorithmController
from core.services.LoggerService import LoggerService
from helpers.TranslationMixin import TranslationMixin
from algorithms.streaming.AIPersonDetector.services import (
    AIPersonStreamingService,
    AIPersonStreamingConfig,
    AIPersonDetection,
)
from algorithms.streaming.AIPersonDetector.views import AIPersonDetectorControlWidget


class AIPersonDetectorController(TranslationMixin, StreamAlgorithmController):
    """Controller for streaming ONNX person detection."""

    def __init__(self, algorithm_config: Dict[str, Any], theme: str, parent=None):
        super().__init__(algorithm_config, theme, parent)
        self.logger = LoggerService()
        self.provides_custom_rendering = True

        # IMPORTANT: no parent so this QObject can move to the worker thread.
        self.person_detector = AIPersonStreamingService(parent=None)
        self.person_detector.performanceUpdate.connect(self._on_performance_update)

        if hasattr(self, "control_widget"):
            self._on_config_changed(self.control_widget.get_config())

        self.detection_count = 0

    def setup_ui(self):
        """Build controls for the streaming algorithm."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        self.control_widget = AIPersonDetectorControlWidget()
        self.control_widget.configChanged.connect(self._on_config_changed)
        layout.addWidget(self.control_widget)

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        """Process frame and emit detections plus rendered frame."""
        try:
            annotated, detections, _timings = self.person_detector.process_frame(frame, timestamp)
            detection_dicts = self._to_detection_dicts(detections)
            self.detection_count += len(detection_dicts)
            self.detectionsReady.emit(detection_dicts)
            self.frameProcessed.emit(annotated)
            return detection_dicts
        except Exception as exc:
            self.logger.error(f"AIPersonDetectorController frame processing failed: {exc}")
            self.frameProcessed.emit(frame.copy())
            return []

    @Slot(dict)
    def _on_config_changed(self, config: Dict[str, Any]):
        """Apply widget config to service."""
        service_config = self._to_service_config(config)
        self.person_detector.update_config(service_config)
        self._emit_config_changed()

    @Slot(dict)
    def _on_performance_update(self, metrics: dict):
        """Forward performance information to status line."""
        fps = float(metrics.get("fps", 0.0))
        processing_ms = float(metrics.get("avg_processing_time_ms", 0.0))
        self._emit_status(
            self.tr("FPS: {fps} | Processing: {ms}ms").format(
                fps=f"{fps:.1f}",
                ms=f"{processing_ms:.1f}",
            )
        )

    def _to_service_config(self, config: Dict[str, Any]) -> AIPersonStreamingConfig:
        """Convert UI config dictionary to service config dataclass."""
        if "confidence_threshold" in config:
            confidence = float(config.get("confidence_threshold", 0.5))
        else:
            confidence_percent = float(config.get("person_detector_confidence", 50))
            confidence = confidence_percent / 100.0

        confidence = max(0.01, min(1.0, confidence))
        render_shape = int(config.get("render_shape", 0))
        max_detections = int(
            config.get(
                "max_detections",
                config.get("max_detections_to_render", 25),
            )
        )
        show_labels = bool(config.get("show_labels", config.get("render_text", True)))

        return AIPersonStreamingConfig(
            confidence_threshold=confidence,
            cpu_only=bool(config.get("cpu_only", False)),
            high_resolution_model=bool(config.get("high_resolution_model", False)),
            show_labels=show_labels,
            max_detections=max_detections,
            processing_width=int(config.get("processing_width", 99999)),
            processing_height=int(config.get("processing_height", 99999)),
            target_fps=int(config.get("target_fps", 0)),
            render_shape=max(0, min(3, render_shape)),
        )

    def _to_detection_dicts(self, detections: List[AIPersonDetection]) -> List[Dict]:
        """Convert detection objects to viewer-friendly dictionaries."""
        output = []
        for detection in detections:
            output.append(
                {
                    "bbox": detection.bbox,
                    "centroid": detection.centroid,
                    "area": detection.area,
                    "confidence": detection.confidence,
                    "class_name": "Person",
                    "detection_type": detection.detection_type,
                    "timestamp": detection.timestamp,
                    "metadata": detection.metadata,
                }
            )
        return output

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration from controls."""
        return self.control_widget.get_config()

    def set_config(self, config: Dict[str, Any]):
        """Apply configuration into controls/service."""
        self.control_widget.set_config(config)
        self._on_config_changed(self.control_widget.get_config())
