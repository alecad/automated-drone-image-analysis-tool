import os
import sys

sys.path.append('app')
from core.services.AOILocationService import AOILocationService


def test_compute_aoi_coordinates():
    service = AOILocationService('resources/dem')
    result = service.compute_aoi_coordinates('resources/img/DJI_20240908125346_0045_V.JPG', 2000, 1500)
    assert result is not None
    lat, lon, alt = result
    assert 39.34 < lat < 39.35
    assert 8.40 < lon < 8.41
    assert alt > 0
