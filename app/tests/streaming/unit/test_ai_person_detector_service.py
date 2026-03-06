"""Unit tests for streaming AI person detector service."""

import numpy as np
from unittest.mock import Mock, patch

from algorithms.streaming.AIPersonDetector.services.AIPersonStreamingService import (
    AIPersonStreamingService,
    AIPersonStreamingConfig,
)


class TestAIPersonStreamingService:
    """Test suite for AIPersonStreamingService."""

    def test_process_frame_converts_raw_boxes_to_detections(self):
        """Service should convert model boxes into streaming detection objects."""
        service = AIPersonStreamingService()
        service.update_config(
            AIPersonStreamingConfig(
                confidence_threshold=0.5,
                max_detections=10,
                show_labels=False,
            )
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch(
            "algorithms.streaming.AIPersonDetector.services.AIPersonStreamingService.ONNXRUNTIME_AVAILABLE",
            True,
        ), patch.object(
            service,
            "_infer",
            return_value=[(10, 20, 110, 220, 0.9), (0, 0, 2, 2, 0.4)],
        ):
            annotated, detections, timings = service.process_frame(frame, 12.34)

        assert annotated.shape == frame.shape
        assert len(detections) == 1
        assert detections[0].bbox == (10, 20, 100, 200)
        assert detections[0].detection_type == "person"
        assert detections[0].confidence == 0.9
        assert timings.total_ms >= 0.0

    def test_process_frame_returns_empty_when_runtime_unavailable(self):
        """Service should fail gracefully when ONNX runtime is unavailable."""
        service = AIPersonStreamingService()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        with patch(
            "algorithms.streaming.AIPersonDetector.services.AIPersonStreamingService.ONNXRUNTIME_AVAILABLE",
            False,
        ):
            annotated, detections, timings = service.process_frame(frame, 1.0)

        assert annotated.shape == frame.shape
        assert detections == []
        assert timings.total_ms >= 0.0

    def test_get_session_uses_cpu_fallback(self):
        """Session creation should fall back to CPU provider after provider failure."""
        service = AIPersonStreamingService()
        cfg = AIPersonStreamingConfig(cpu_only=False)

        mock_ort = Mock()
        mock_ort.ExecutionMode.ORT_SEQUENTIAL = object()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = object()
        mock_ort.SessionOptions.return_value = Mock()
        cpu_session = Mock()
        mock_ort.InferenceSession.side_effect = [RuntimeError("DML unavailable"), cpu_session]

        with patch(
            "algorithms.streaming.AIPersonDetector.services.AIPersonStreamingService.ONNXRUNTIME_AVAILABLE",
            True,
        ), patch(
            "algorithms.streaming.AIPersonDetector.services.AIPersonStreamingService.ort",
            mock_ort,
        ), patch.object(
            service,
            "_resolve_model_path",
            return_value="/tmp/model.onnx",
        ):
            session = service._get_session(cfg)

        assert session is cpu_session
        assert mock_ort.InferenceSession.call_count == 2
