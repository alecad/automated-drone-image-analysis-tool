import os
from core.services.AOIGeolocationService import AOIGeolocationService


def test_aoi_geolocation_center():
    dem_folder = os.path.abspath('resources/dem')
    image_path = os.path.abspath('resources/img/DJI_20240908125346_0045_V.JPG')
    service = AOIGeolocationService(dem_folder)
    result = service.compute_aoi_coordinate(image_path, (2000, 1500))
    assert result is not None
    lat, lon, alt = result
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
    assert isinstance(alt, float)
