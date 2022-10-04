import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

from cols.cols import get_ride_cols


def compute_climb_metrics(climb, ride):
    """Compute some metrics to caracterize a climb
    
    elevation(m) -> max elevation of the climb (likely to be at the end)
    distance(m) -> distance of the climb
    elevation_gain(m) -> difference between highest and lowest point (raw)
    difficulty(pts) -> centcols squared cotation method computing the difficulty of a climb
    slope(%) -> average slope of the climb
    """
    # centcols squared cotation method
    ride_climb = ride.iloc[climb['beginning']:climb['end']]
    metrics = {
        'elevation':  int(ride_climb['elevation'].max()),
        'distance': int(ride_climb['total_distance'].max() - ride_climb['total_distance'].min()),
        'elevation_gain': int(ride_climb['elevation'].max() - ride_climb['elevation'].min()),
        'difficulty': int(10 * ride_climb['climb_difficulty'].sum())
    }
    metrics['slope'] = round(100 * metrics['elevation_gain'] / metrics['distance'], 1)
    return pd.Series(metrics)


def detect_ride_climbs(ride):
    """Detect the climbs of the ride"""

    # Use elevation difference as a proxy for how much to smooth climb points
    points_gap = ride['point_duration'].median()
    sigma = (ride['elevation'].max() - ride['elevation'].min()) ** 0.5 / points_gap

    # Use climb difficulty to help detect climbs more precisely than elevation
    ride['lowered_climb'] = ride['climb_difficulty'] - ride['climb_difficulty'].mean() / 4
    smooth_climb = gaussian_filter1d(ride['lowered_climb'], sigma)

    # Find switching points (start and end of climbs)
    beginning_points = np.where(np.diff(np.sign(smooth_climb)) > 0)[0]
    end_points = np.where(np.diff(np.sign(smooth_climb)) < 0)[0]

    # Make sure we have as much beginning points as end points
    if len(end_points) < len(beginning_points):
        beginning_points = beginning_points[:-1]
    if len(beginning_points) < len(end_points):
        end_points = end_points[1:]

    # Use points of the ride as switching points
    beginning_points = [ride[ride.index <= bp].index.max() for bp in beginning_points]
    end_points = [ride[ride.index <= ep].index.max() for ep in end_points]

    # Put climb in DataFrame and compute their metrics
    climbs = pd.DataFrame(zip(beginning_points, end_points), columns=['beginning', 'end'])
    climbs_metrics = climbs.apply(compute_climb_metrics, ride=ride, axis=1, result_type='reduce')
    return climbs.join(climbs_metrics)


def merge_climbs(climbs, ride, ratio=10):
    """Merge differents parts of climbs detected as several climb
    
    Even if some famous climbs have flat or downhill portions in them, they are considered
    as a unique climb. For instance, Col de la Croix de Fer has two significants (aroud 2km)
    downhill sections but is still considered in its integrality.
    Adjusting the gaussian filtering to do so would lead to very unprecise detection of the
    beginning and end of the climbs, so we need a way to perform a merge.
    The criterium we used is based on the easiest of the two difficulty in one hand, and the
    distance of the non climb section in the other hand.
    We set arbitraly the ratio at a value of 10, meaning the limit case of a merge would be a
    climb such as (2km@7% + 1km@0% + 2km@7%)
    """
    for i in range(len(climbs) - 1):
        climb_1, climb_2 = climbs.loc[i], climbs.loc[i+1]
    
        # Use non climb distance for how much climb should be splitted
        pause_beginning = ride.loc[climb_1['end']]['total_distance']
        pause_end = ride.loc[climb_2['beginning']]['total_distance']
        pause_distance = pause_end - pause_beginning

        # Use the lowest difficulty for how much the climbs are big, and should be reunited
        difficulty = min(climb_1['difficulty'], climb_2['difficulty'])

        if pause_distance <= ratio * difficulty:
            climb = pd.Series([climb_1['beginning'], climb_2['end']], index=['beginning', 'end']).astype(int)
            climb = pd.concat([climb, compute_climb_metrics(climb, ride)])
            # Delete first climb and replace second with merged, so it can be merged again
            climbs = climbs.drop(index=i)
            climbs.loc[i+1] = climb

    return climbs


def filter_climbs(climbs, min_score=20, ratio=0.05):
    """Filter unsignificant climbs.
    
    We filter out very easy climbs which would not be noticed, and small climbs if the ride
    contains some very big one.
    """
    ratio_min_score = ratio * climbs['difficulty'].sum()
    return climbs[climbs['difficulty'] > max(min_score, ratio_min_score)]


def associate_climb_with_col(climb, cols, threshold=100):
    """Associate the climbs with the col detected on the route."""
    cols['distance_from_climb'] = (cols['point_of_ride'] - climb['end']).apply(abs)
    climb['col_name'] = None

    if not cols.empty:
        col = cols.sort_values('distance_from_climb').iloc[0]
        if col['distance_from_climb'] <= threshold:
            climb['col_name'] = col['name']
            climb['elevation'] = col['elevation']

    return climb


def plot_climb(climb, ride):
    """Plot the climbs of the graph"""
    ride_climb = ride.iloc[int(climb['beginning']):int(climb['end'])]
    col_name = climb['col_name'] if climb['col_name'] is not None else 'Climb'
    plt.plot(ride_climb['total_distance'], ride_climb['roll_elevation'], label=col_name)
    end_point = ride_climb.iloc[-1]
    plt.plot(end_point['total_distance'], end_point['elevation'], marker='x', color=plt.gca().lines[-1].get_color())



def plot_col(col, climbs):
    """Plot the cols on the graps"""
    col_name = col['name']
    if col_name not in climbs['col_name'].to_list():
        plt.plot(col['distance_of_ride'], col['elevation'], marker='x', label=col_name)


def list_ride_climbs(ride, plot=True):
    cols = get_ride_cols(ride)
    climbs = detect_ride_climbs(ride)
    climbs = merge_climbs(climbs, ride)
    climbs = filter_climbs(climbs)
    climbs = climbs.apply(associate_climb_with_col, cols=cols, axis=1, result_type='reduce')

    if plot:
        # plot results
        plt.plot(ride['total_distance'], ride['roll_elevation'], label='Elevation')

        for _, climb in climbs.iterrows():
            plot_climb(climb, ride)

        for _, col in cols.iterrows():
            plot_col(col, climbs)

        plt.legend(bbox_to_anchor=(1, 1))

    return climbs
