import pytest
import numpy as np
import tempfile
import os
from algorithms.images.MRMap.services.MRMapService import MRMapService
from algorithms.AlgorithmService import AnalysisResult


@pytest.fixture
def mrmap_service():
    """Fixture providing an MRMapService instance."""
    options = {
        'threshold': 95,
        'segments': 2,
        'window': 3,
        'colorspace': 'RGB'
    }
    return MRMapService(
        identifier=(255, 0, 0),
        min_area=10,
        max_area=1000,
        aoi_radius=5,
        combine_aois=True,
        options=options
    )


@pytest.fixture
def test_image():
    """Create a test image."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    return img


def test_mrmap_service_initialization(mrmap_service):
    """Test MRMapService initialization."""
    assert mrmap_service.name == 'MRMap'
    assert mrmap_service.segments == 2
    assert mrmap_service.threshold == 95
    assert mrmap_service.window_size == 3
    assert mrmap_service.colorspace == 'RGB'


def test_mrmap_service_hsv_colorspace():
    """Test MRMapService with HSV colorspace."""
    options = {
        'threshold': 95,
        'segments': 2,
        'window': 3,
        'colorspace': 'HSV'
    }
    service = MRMapService(
        identifier=(255, 0, 0),
        min_area=10,
        max_area=1000,
        aoi_radius=5,
        combine_aois=True,
        options=options
    )
    assert service.colorspace == 'HSV'


def test_mrmap_service_lab_colorspace():
    """Test MRMapService with LAB colorspace."""
    options = {
        'threshold': 95,
        'segments': 2,
        'window': 3,
        'colorspace': 'LAB'
    }
    service = MRMapService(
        identifier=(255, 0, 0),
        min_area=10,
        max_area=1000,
        aoi_radius=5,
        combine_aois=True,
        options=options
    )
    assert service.colorspace == 'LAB'


def test_process_image(mrmap_service, test_image):
    """Test processing an image with MRMap algorithm."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = tmpdir
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(input_dir, "test.jpg")

        result = mrmap_service.process_image(test_image, full_path, input_dir, output_dir)

        assert isinstance(result, AnalysisResult)
        assert result.input_path == full_path
        assert result.error_message is None


def test_mask_contains_only_flagged_pixels():
    """Mask must contain only actually flagged pixels, not filled bounding rectangles."""
    options = {'threshold': 95, 'segments': 1, 'window': 5, 'colorspace': 'RGB'}
    service = MRMapService(
        identifier=(255, 0, 0),
        min_area=3,
        max_area=0,
        aoi_radius=0,
        combine_aois=False,
        options=options,
    )

    pixel_anom = np.zeros((50, 50), dtype=bool)
    sparse_coords = [(10, 10), (12, 10), (14, 12), (10, 14), (14, 14)]
    for x, y in sparse_coords:
        pixel_anom[y, x] = True

    mask, clusters = service._getMRMapsContours(pixel_anom)

    assert len(clusters) == 1
    assert len(clusters[0]['pixels']) == len(sparse_coords)
    assert int(mask.sum() // 255) == len(sparse_coords)
    for x, y in sparse_coords:
        assert mask[y, x] == 255

    bounding_area = (14 - 10 + 1) * (14 - 10 + 1)
    assert int(mask.sum() // 255) < bounding_area


def test_build_aois_from_scattered_cluster():
    """A scattered BFS cluster must produce exactly one AOI with actual pixels."""
    options = {'threshold': 95, 'segments': 1, 'window': 5, 'colorspace': 'RGB'}
    service = MRMapService(
        identifier=(255, 0, 0),
        min_area=3,
        max_area=0,
        aoi_radius=2,
        combine_aois=False,
        options=options,
    )

    sparse_coords = [(10, 10), (12, 10), (14, 12), (10, 14), (14, 14)]
    pixel_anom = np.zeros((50, 50), dtype=bool)
    for x, y in sparse_coords:
        pixel_anom[y, x] = True

    mask, clusters = service._getMRMapsContours(pixel_anom)
    aois, base_count = service._build_aois_from_clusters(clusters, (50, 50, 3))

    assert base_count == 1
    assert len(aois) == 1
    aoi = aois[0]
    assert aoi['area'] == len(sparse_coords)
    assert len(aoi['detected_pixels']) == len(sparse_coords)
    returned = {tuple(p) for p in aoi['detected_pixels']}
    expected = set(sparse_coords)
    assert returned == expected
    assert len(aoi['contour']) == 4  # rectangle corners


def test_add_confidence_scores(mrmap_service):
    """Test adding confidence scores to AOIs."""
    areas_of_interest = [
        {
            'center': (50, 50),
            'radius': 10,
            'detected_pixels': [(45, 45), (46, 46), (47, 47)]
        }
    ]

    bin_counts = np.zeros((100, 100), dtype=np.float32)
    bin_counts[45:48, 45:48] = 5.0  # Low bin count = rare = anomaly

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:48, 45:48] = 1

    result = mrmap_service._add_confidence_scores(
        areas_of_interest, bin_counts, mask
    )

    assert len(result) == 1
    assert 'confidence' in result[0]
    assert 'score_type' in result[0]
    assert result[0]['score_type'] == 'rarity'
