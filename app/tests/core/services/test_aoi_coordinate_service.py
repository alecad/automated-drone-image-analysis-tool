import math

from core.services.AOICoordinateService import AOICoordinateService
from core.services.KMLGeneratorService import KMLGeneratorService


def _haversine(lat1, lon1, lat2, lon2):
    """Return distance in meters between two lat/lon pairs."""
    R = 6378137.0
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def test_example_aoi_coordinate():
    service = AOICoordinateService('app/dem/output_SRTMGL1.tif')
    lat, lon = service.pixel_to_latlon('app/example_img/DJI_20240908125346_0045_V.JPG', 3420, 1180)
    expected_lat = 39.341413
    expected_lon = 8.404803
    distance = _haversine(lat, lon, expected_lat, expected_lon)
    assert distance <= 50


def test_second_aoi_coordinate():
    service = AOICoordinateService('app/dem/output_SRTMGL1.tif')
    lat, lon = service.pixel_to_latlon('app/example_img/DJI_20240908125346_0045_V.JPG', 572, 2740)
    expected_lat = 39.342239
    expected_lon = 8.403548
    distance = _haversine(lat, lon, expected_lat, expected_lon)
    assert distance <= 50


def test_pixel_to_view_and_kml(tmp_path):
    service = AOICoordinateService('app/dem/output_SRTMGL1.tif')
    view = service.pixel_to_view('app/example_img/DJI_20240908125346_0045_V.JPG', 3420, 1180)
    assert view is not None
    assert 0 <= view['heading'] <= 360
    assert 0 <= view['tilt'] <= 90
    assert view['range'] > 0

    kml_service = KMLGeneratorService()
    out = tmp_path / 'view.kml'
    kml_service.generate_view_kml(
        view['lat'], view['lon'], view['heading'], view['tilt'], view['range'], str(out)
    )
    content = out.read_text()
    assert '<LookAt>' in content
    assert '<heading>' in content
