import pandas as pd
from gpxcsv import gpxtolist
import gzip
import shutil


COLUMNS_NAMES = {
    'lat': 'latitude',
    'lon': 'longitude',
    'hr': 'heart_rate',
    'ele': 'elevation',
    'time':	'time'
}


def load(filename, is_compressed):
    """Load a file in a readable DataFrame"""
    if is_compressed:
        with gzip.open(f'gpx_files/{filename}.gpx.gz', 'rb') as f_compressed:
            with open(f'gpx_files/{filename}.gpx', 'wb') as f_readable:
                shutil.copyfileobj(f_compressed, f_readable)

    ride = pd.DataFrame(gpxtolist(f'gpx_files/{filename}.gpx'))
    ride['time'] = ride['time'].apply(pd.to_datetime)
    ride = ride.rename(columns=COLUMNS_NAMES)
    columns = [column for column in ride.columns if column in COLUMNS_NAMES.values()]
    return ride[columns]
