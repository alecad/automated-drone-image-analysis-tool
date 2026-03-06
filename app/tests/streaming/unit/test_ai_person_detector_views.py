"""Unit tests for streaming AI Person Detector control widget."""

from algorithms.streaming.AIPersonDetector.views.AIPersonDetectorControlWidget import (
    AIPersonDetectorControlWidget,
)


class TestAIPersonDetectorControlWidget:
    """Test suite for AIPersonDetectorControlWidget."""

    def test_initialization_creates_expected_tabs(self, qapp):
        """Widget should use tabbed layout like other streaming algorithms."""
        widget = AIPersonDetectorControlWidget()

        assert widget.tabs.count() == 4
        assert widget.tabs.tabText(0) == "Person Detection"
        assert widget.tabs.tabText(1) == "Input && Processing"
        assert widget.tabs.tabText(2) == "Frame"
        assert widget.tabs.tabText(3) == "Rendering && Cleanup"

    def test_get_config_includes_shared_container_settings(self, qapp):
        """Configuration should include shared input/frame/rendering settings."""
        widget = AIPersonDetectorControlWidget()
        config = widget.get_config()

        assert "person_detector_confidence" in config
        assert "processing_width" in config
        assert "processing_height" in config
        assert "target_fps" in config
        assert "render_shape" in config
        assert "render_text" in config
        assert "mask_enabled" in config
        assert "max_detections" in config

    def test_set_config_maps_legacy_fields(self, qapp):
        """Legacy fields should map into shared rendering/input controls."""
        widget = AIPersonDetectorControlWidget()
        widget.set_config(
            {
                "person_detector_confidence": 70,
                "cpu_only": True,
                "high_resolution_model": True,
                "show_labels": False,
                "max_detections": 5,
                "target_fps": 15,
                "processing_width": 99999,
                "processing_height": 99999,
            }
        )

        config = widget.get_config()
        assert config["person_detector_confidence"] == 70
        assert config["cpu_only"] is True
        assert config["high_resolution_model"] is True
        assert config["show_labels"] is False
        assert config["max_detections"] == 5
        assert config["target_fps"] == 15
