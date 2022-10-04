import numpy as np

from geopy import distance
from scipy.ndimage import gaussian_filter1d


def calculate_intermediate_distance(point):
    return distance.distance(
        (point.latitude_shift, point.longitude_shift),
        (point.latitude, point.longitude)
    ).m


def get_distance(ride):
    """Get distance metrics of a ride"""
    ride['latitude_shift'] = ride['latitude'].shift().fillna(ride['latitude'][0])
    ride['longitude_shift'] = ride['longitude'].shift().fillna(ride['longitude'][0])
    ride['point_distance'] = ride.apply(calculate_intermediate_distance, axis=1)
    ride['total_distance'] = ride['point_distance'].cumsum()
    ride = ride[ride['point_distance'] > 0]
    return ride.drop(columns=['latitude_shift', 'longitude_shift'])


def get_duration(ride):
    """Get duration metrics of a ride"""
    ride['point_duration'] = ride['time'].diff().apply(lambda p: p.seconds)
    ride['total_duration'] = ride['point_duration'].cumsum()
    return ride


def get_elevation_gain(ride):
    """Get elevation gain metrics of a ride"""
    # Some device only have int for elevation, smoother it to have more precise results
    ride['roll_elevation'] = gaussian_filter1d(ride['elevation'], 10)
    ride['point_elevation_diff'] = ride['roll_elevation'].diff()
    ride['point_elevation_gain'] = ride['point_elevation_diff'].apply(lambda point: max (0, point))
    ride['point_elevation_loss'] = ride['point_elevation_diff'].apply(lambda point: max (0, -point))
    ride['total_elevation_diff'] = ride['point_elevation_diff'].cumsum()
    ride['total_elevation_gain'] = ride['point_elevation_gain'].cumsum()
    ride['total_elevation_loss'] = ride['point_elevation_loss'].cumsum()
    return ride


def get_climb(ride):
    """Get climb metrics of a ride"""
    ride['slope'] = (ride['point_elevation_diff'] / ride['point_distance']).apply(lambda s: min(max(s, -0.2), 0.2))
    ride['climb_difficulty'] = ride['slope'] * ride['point_elevation_gain']
    return ride


def get_speeds(ride):
    """Get speed metrics of a ride"""
    ride['speed'] = ride['point_distance'] / ride['point_duration']
    ride['vertical_speed'] = ride['point_elevation_diff'] / ride['point_duration']
    return ride


def get_difficulty(ride):
    """Get difficulty metrics of a ride"""
    # Assign a difficulty score to each of the points
    ride['point_difficulty'] = 0.001 * np.exp(20 * ride['slope']) * ride['point_distance']
    ride['total_difficulty'] = ride['point_difficulty'].cumsum()
    return ride


def treat_ride(ride):
    """Get all the metics of a ride"""
    ride = get_distance(ride)
    ride = get_duration(ride)
    ride = get_elevation_gain(ride)
    ride = get_climb(ride)
    ride = get_speeds(ride)
    ride = get_difficulty(ride)

    return ride
