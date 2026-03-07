"""
Comprehensive tests for AnalyzeService.

Tests the core image analysis orchestration service.
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PySide6.QtCore import QObject
from core.services.AnalyzeService import AnalyzeService


@pytest.fixture
def analyze_service():
    """Fixture providing an AnalyzeService instance."""
    algorithm = {
        'name': 'ColorRange',
        'type': 'RGB',
        'service': 'ColorRangeService'
    }
    options = {
        'color_ranges': [
            {'color_range': [(100, 150, 200), (120, 170, 220)]}
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'input')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        service = AnalyzeService(
            id=1,
            algorithm=algorithm,
            input=input_dir,
            output=output_dir,
            identifier_color=(100, 150, 200),
            min_area=10,
            num_processes=1,
            max_aois=100,
            aoi_radius=5,
            histogram_reference_path=None,
            kmeans_clusters=None,
            options=options,
            max_area=1000,
            processing_resolution=1.0
        )
        yield service


def test_analyze_service_initialization(analyze_service):
    """Test AnalyzeService initialization."""
    assert analyze_service.algorithm is not None
    assert analyze_service.min_area == 10
    assert analyze_service.max_area == 1000
    assert analyze_service.num_processes == 1
    assert analyze_service.processing_resolution == 1.0


def test_analyze_service_thermal_detection():
    """Test that thermal algorithms are properly detected."""
    algorithm = {
        'name': 'ThermalRange',
        'type': 'Thermal',
        'service': 'ThermalRangeService'
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'input')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        service = AnalyzeService(
            id=1,
            algorithm=algorithm,
            input=input_dir,
            output=output_dir,
            identifier_color=(255, 0, 0),
            min_area=10,
            num_processes=1,
            max_aois=100,
            aoi_radius=5,
            histogram_reference_path=None,
            kmeans_clusters=None,
            options={'minTemp': 20.0, 'maxTemp': 30.0},
            max_area=1000,
            processing_resolution=1.0
        )

        assert service.is_thermal is True


def test_analyze_service_processing_resolution():
    """Test processing resolution scaling."""
    algorithm = {
        'name': 'ColorRange',
        'type': 'RGB',
        'service': 'ColorRangeService'
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'input')
        output_dir = os.path.join(tmpdir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        service = AnalyzeService(
            id=1,
            algorithm=algorithm,
            input=input_dir,
            output=output_dir,
            identifier_color=(100, 150, 200),
            min_area=10,
            num_processes=1,
            max_aois=100,
            aoi_radius=5,
            histogram_reference_path=None,
            kmeans_clusters=None,
            options={},
            max_area=1000,
            processing_resolution=0.5  # 50% resolution
        )

        assert service.processing_resolution == 0.5


def test_analyze_service_signals(analyze_service):
    """Test that signals are properly defined."""
    assert hasattr(analyze_service, 'sig_msg')
    assert hasattr(analyze_service, 'sig_aois')
    assert hasattr(analyze_service, 'sig_done')


def test_analyze_service_cancellation(analyze_service):
    """Test cancellation functionality."""
    assert analyze_service.cancelled is False
    analyze_service.cancelled = True
    assert analyze_service.cancelled is True


@patch('core.services.AnalyzeService.cv2.imdecode')
@patch('core.services.AnalyzeService.np.fromfile')
def test_process_file_unknown_algorithm_service_returns_error(mock_fromfile, mock_imdecode):
    """Unknown algorithm services should return an explicit error result."""
    mock_fromfile.return_value = np.array([1], dtype=np.uint8)
    mock_imdecode.return_value = np.zeros((20, 20, 3), dtype=np.uint8)

    algorithm = {
        'name': 'ColorRange',
        'type': 'RGB',
        'service': 'MissingService',
        'combine_overlapping_aois': True
    }

    result = AnalyzeService.process_file(
        algorithm=algorithm,
        identifier_color=(100, 150, 200),
        min_area=10,
        max_area=1000,
        aoi_radius=5,
        options={},
        full_path='/tmp/fake.jpg',
        input_dir='/tmp/input',
        output_dir='/tmp/output',
        hist_ref_path=None,
        kmeans_clusters=None,
        thermal=False,
        processing_resolution=1.0
    )

    assert result is not None
    assert result.input_path == '/tmp/fake.jpg'
    assert result.error_message is not None
    assert 'Unknown algorithm service: MissingService' in result.error_message
