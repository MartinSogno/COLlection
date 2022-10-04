import pandas as pd
from geopy import distance

from cols.cent_cols import get_all_cols

def restrict_to_possible_cols(ride, cols, margin=0.01):
    """Restrict the list of cols to the one in the ride circumscribed square"""
    return cols[
        (cols['latitude'] > ride['latitude'].min() - margin)
        & (cols['latitude'] < ride['latitude'].max() + margin)
        & (cols['longitude'] > ride['longitude'].min() - margin)
        & (cols['longitude'] < ride['longitude'].max() + margin)   
    ]


def get_destination(coordinates, bearing, length):
    return distance.distance(meters=length).destination(coordinates, bearing)


def is_col_on_ride(col, ride, threshold=30):
    """Check wether a col is on a ride. 
    
    Threshold(m) is the maximum distance from the ride we allow."""
    coordinates = (col['latitude'], col['longitude'])
    close_to_col = ride[
        (ride['latitude'] <  get_destination(coordinates, 0, threshold).latitude)
        & (ride['latitude'] >  get_destination(coordinates, 180, threshold).latitude)
        & (ride['longitude'] <  get_destination(coordinates, 90, threshold).longitude)
        & (ride['longitude'] >  get_destination(coordinates, 270, threshold).longitude)   
    ]
    return False if close_to_col.empty else close_to_col.iloc[0].name


def get_ride_cols(ride):
    """Get the cols present on a ride."""
    cols = get_all_cols()
    possible_cols = restrict_to_possible_cols(ride, cols.copy())
    possible_cols['point_of_ride'] = possible_cols.apply(lambda c: is_col_on_ride(c, ride), axis=1, result_type='reduce')
    cols = possible_cols[possible_cols['point_of_ride'] != False]
    return cols[['name', 'elevation', 'point_of_ride']].sort_values('point_of_ride')
