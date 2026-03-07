"""Streaming AI person detector service backed by ONNX Runtime."""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple
import time
import cv2
import numpy as np
import sys
import os

from PySide6.QtCore import QObject, Signal

from core.services.LoggerService import LoggerService
from core.services.streaming.StreamingUtils import StageTimings

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    ort = None
    ONNXRUNTIME_AVAILABLE = False


@dataclass
class AIPersonStreamingConfig:
    """Configuration for streaming AI person detection."""
    confidence_threshold: float = 0.50
    cpu_only: bool = False
    high_resolution_model: bool = False
    show_labels: bool = True
    max_detections: int = 25
    processing_width: int = 99999
    processing_height: int = 99999
    target_fps: int = 0
    render_shape: int = 0  # 0=Box, 1=Circle, 2=Dot, 3=Off


@dataclass
class AIPersonDetection:
    """Container for a person detection."""
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    area: float
    confidence: float
    timestamp: float
    detection_type: str = "person"
    metadata: Dict = field(default_factory=dict)


class AIPersonStreamingService(QObject):
    """Run ONNX person detection on streaming frames."""

    performanceUpdate = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = LoggerService()
        self._config = AIPersonStreamingConfig()
        self._config_lock = Lock()
        self._session_cache = {}
        self._timings: List[float] = []
        self._max_timing_samples = 60

    def update_config(self, config: AIPersonStreamingConfig):
        """Update detector configuration thread-safely."""
        with self._config_lock:
            self._config = config

    def get_config(self) -> AIPersonStreamingConfig:
        """Return current immutable config snapshot."""
        with self._config_lock:
            return AIPersonStreamingConfig(**self._config.__dict__)

    def process_frame(self, frame: np.ndarray, timestamp: float):
        """Process one frame and return annotated frame, detections, timings."""
        timings = StageTimings()
        total_start = time.time()
        try:
            if frame is None or frame.size == 0:
                timings.total_ms = (time.time() - total_start) * 1000.0
                return frame, [], timings

            cfg = self.get_config()

            if not ONNXRUNTIME_AVAILABLE:
                timings.total_ms = (time.time() - total_start) * 1000.0
                return frame.copy(), [], timings

            pre_start = time.time()
            processing_frame, scale_x, scale_y = self._prepare_processing_frame(frame, cfg)
            timings.preprocessing_ms = (time.time() - pre_start) * 1000.0

            detect_start = time.time()
            raw_boxes = self._infer(processing_frame, cfg)
            timings.detection_ms = (time.time() - detect_start) * 1000.0

            detections = self._to_detection_objects(
                raw_boxes=raw_boxes,
                scale_x=scale_x,
                scale_y=scale_y,
                timestamp=timestamp,
                confidence_threshold=cfg.confidence_threshold,
                max_detections=cfg.max_detections
            )

            render_start = time.time()
            annotated = self._annotate_frame(frame, detections, cfg)
            timings.render_ms = (time.time() - render_start) * 1000.0

            timings.total_ms = (time.time() - total_start) * 1000.0
            self._record_performance(timings.total_ms, len(detections))
            return annotated, detections, timings
        except Exception as exc:
            self.logger.error(f"AIPerson streaming frame processing failed: {exc}")
            timings.total_ms = (time.time() - total_start) * 1000.0
            return frame.copy(), [], timings

    def _prepare_processing_frame(self, frame: np.ndarray, cfg: AIPersonStreamingConfig):
        """Resize frame for detection if processing resolution is configured."""
        src_h, src_w = frame.shape[:2]
        target_w = int(cfg.processing_width) if cfg.processing_width else src_w
        target_h = int(cfg.processing_height) if cfg.processing_height else src_h

        if target_w >= 99999 or target_h >= 99999:
            return frame, 1.0, 1.0

        target_w = max(10, min(target_w, src_w))
        target_h = max(10, min(target_h, src_h))
        if target_w == src_w and target_h == src_h:
            return frame, 1.0, 1.0

        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scale_x = float(src_w) / float(target_w)
        scale_y = float(src_h) / float(target_h)
        return resized, scale_x, scale_y

    def _infer(self, frame_bgr: np.ndarray, cfg: AIPersonStreamingConfig):
        """Run ONNX inference and return raw boxes in frame coordinates."""
        session = self._get_session(cfg)
        input_name = session.get_inputs()[0].name

        model_size = 1024 if cfg.high_resolution_model else 640
        frame_h, frame_w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = session.run(None, {input_name: input_tensor})
        predictions = outputs[0]
        if getattr(predictions, "ndim", 0) == 3:
            predictions = predictions[0]

        raw_boxes = []
        for pred in predictions:
            if len(pred) < 6:
                continue
            x1, y1, x2, y2, conf, cls = pred[:6]
            # Preserve person-class behavior from existing image detector models.
            if int(cls) != 0:
                continue
            bx1 = int((float(x1) / model_size) * frame_w)
            by1 = int((float(y1) / model_size) * frame_h)
            bx2 = int((float(x2) / model_size) * frame_w)
            by2 = int((float(y2) / model_size) * frame_h)
            raw_boxes.append((bx1, by1, bx2, by2, float(conf)))
        return raw_boxes

    def _to_detection_objects(
        self,
        raw_boxes: List[Tuple[int, int, int, int, float]],
        scale_x: float,
        scale_y: float,
        timestamp: float,
        confidence_threshold: float,
        max_detections: int
    ) -> List[AIPersonDetection]:
        """Convert raw boxes to streaming detection objects."""
        detections: List[AIPersonDetection] = []
        for x1, y1, x2, y2, conf in raw_boxes:
            if conf < confidence_threshold:
                continue

            sx1 = int(x1 * scale_x)
            sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x)
            sy2 = int(y2 * scale_y)
            w = max(0, sx2 - sx1)
            h = max(0, sy2 - sy1)
            if w == 0 or h == 0:
                continue

            detections.append(
                AIPersonDetection(
                    bbox=(sx1, sy1, w, h),
                    centroid=(sx1 + (w // 2), sy1 + (h // 2)),
                    area=float(w * h),
                    confidence=float(conf),
                    timestamp=float(timestamp),
                    detection_type="person",
                    metadata={"model": "onnx_ai_person"}
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if max_detections > 0:
            detections = detections[:max_detections]
        return detections

    def _annotate_frame(self, frame: np.ndarray, detections: List[AIPersonDetection], cfg: AIPersonStreamingConfig):
        """Render detections to an output frame."""
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox

            if cfg.render_shape == 0:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.circle(annotated, det.centroid, 2, (0, 200, 255), -1)
            elif cfg.render_shape == 1:
                radius = max(4, int(max(w, h) * 0.5))
                cv2.circle(annotated, det.centroid, radius, (0, 200, 255), 2)
            elif cfg.render_shape == 2:
                cv2.circle(annotated, det.centroid, 4, (0, 200, 255), -1)

            if cfg.show_labels:
                label = f"Person {det.confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (det.centroid[0] - 40, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA
                )
        return annotated

    def _record_performance(self, total_ms: float, detection_count: int):
        """Emit basic performance metrics."""
        self._timings.append(total_ms)
        if len(self._timings) > self._max_timing_samples:
            self._timings.pop(0)
        avg_ms = float(np.mean(self._timings)) if self._timings else 0.0
        fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0
        self.performanceUpdate.emit(
            {
                "fps": fps,
                "avg_processing_time_ms": avg_ms,
                "detections_count": detection_count,
            }
        )

    def _get_session(self, cfg: AIPersonStreamingConfig):
        """Get cached ONNX session for current model/provider settings."""
        model_path = self._resolve_model_path(cfg)
        cache_key = (model_path, bool(cfg.cpu_only))
        cached = self._session_cache.get(cache_key)
        if cached is not None:
            return cached

        if not ONNXRUNTIME_AVAILABLE or ort is None:
            raise RuntimeError("onnxruntime is not available")

        providers = ["CPUExecutionProvider"] if cfg.cpu_only else ["DmlExecutionProvider", "CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = False
        session_options.enable_profiling = False
        session_options.intra_op_num_threads = 1

        try:
            session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        except Exception:
            session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])

        self._session_cache[cache_key] = session
        return session

    def _resolve_model_path(self, cfg: AIPersonStreamingConfig) -> str:
        """Resolve ONNX model path for source and frozen builds."""
        model_name = "ai_person_model_V3_1024.onnx" if cfg.high_resolution_model else "ai_person_model_V3_640.onnx"
        if getattr(sys, "frozen", False):
            return os.path.join(
                sys._MEIPASS,
                "algorithms",
                "models",
                "AIPersonDetector",
                model_name
            )

        model_path = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "AIPersonDetector"
            / model_name
        )
        return str(model_path)
