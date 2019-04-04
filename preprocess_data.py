""" This code preprocesses the flown trajectories and
other relevant data.

Added:
- NCOP and XCOP coordinates.
- WCT category.
- Day of week (DOW)
- Hour of the first observation

All coordinates are converted to XY and the required
amount of points is selected from the flown trajectory.

Filters:
- Flights that are less than 5 minutes from the AOR border
are discarded.
- Flights that have a positive Y entry coordinate are discarded,
since they are not coming from the ROU sector and ended up in the
dataset due to erroneous information in the database.


"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from CoorConv import CoorConv


N_OBSERV = 5  # number of 'observed' points from trajectory --> features
PERIOD = 5  # interval between 'observed' points in seconds

# class for coordinates conversion
conv_coord = CoorConv(51, 8)  # MUAC airspace center: 51 deg lat, 8 deg long

path_to_data = 'data/years'
resources_folder = 'resources'
interim_data_folder = 'interim_data/%d_observ'%N_OBSERV
data_files = os.listdir(path_to_data)


def encode_cyclic(df, col, max_val):
    df[col + '_SIN'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_COS'] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def interpolate_trajectory(trajectory):
    """Linearly interpolate trajectory.
         1. Set time as index.
         2. Convert available points to XY.
         3. Upsample time to 1 second interval.
         4. Linearly interpolate trajectory.

        The result of a linear interpolation was compared to the
        result of interpolating using average speed and yielded small
        distance errors (i.e. average of maximum distance errors
        is 5.3 meters based on 654 flights (flight on July 1st, 2018))."""

    trajectory = trajectory.copy()

    # although the trajectory should be already sorted by time, sort it again
    trajectory = trajectory[['LAT', 'LON', 'TIME_']].sort_values(by='TIME_', ascending=True)

    # set index to time for resampling
    trajectory.set_index('TIME_', inplace=True)

    # convert latitude and longitude to XY
    trajectory[['X', 'Y']] = trajectory.apply(lambda row: pd.Series(conv_coord.lalo_to_xy(row['LAT'], row['LON'])),
                                              axis=1)
    # upsample time and linearly interpolate coordinates
    trajectory_interp = trajectory.resample('S').mean().interpolate()

    # set time back as a column
    trajectory_interp['TIME_'] = trajectory_interp.index
    return trajectory_interp


def prepare_trajectory(trajectory):
    """ Prepare trajectory to be used as a predictor.

    Upsample trajectory to contain points every second and
    linearly interpolate points' coordinates.

    After points are interpolated,
    select first N_OBSERV points with the interval of PERIOD seconds.
    This is done to resemble the update rate of the surveillance data
    which is updated in cycles equal or close to a pattern of
    5-5-5-5-4 seconds.

    ToDo Return None if trajectory contains less than 1 minute of flight
    """

    trajectory_interp = interpolate_trajectory(trajectory)

    # visually inspect interpolation
    # visualise_trajectory(trajectory_interp, points)  # this line was used when interpolating using average speed

    # select first N_OBSERV points spaced at PERIOD
    features_x = trajectory_interp['X'].values.tolist()[:N_OBSERV * PERIOD:PERIOD]
    features_y = trajectory_interp['Y'].values.tolist()[:N_OBSERV * PERIOD:PERIOD]
    features_t = pd.Series(trajectory_interp.index.values)[:N_OBSERV * PERIOD:PERIOD]
    features = list(zip(features_x, features_y, features_t))
    features = [ft for fts in features for ft in fts]  # flatten list

    features_names_x = ['X%d' % n for n in range(1, N_OBSERV + 1, 1)]
    features_names_y = ['Y%d' % n for n in range(1, N_OBSERV + 1, 1)]
    features_names_t = ['T%d' % n for n in range(1, N_OBSERV + 1, 1)]
    features_names = list(zip(features_names_x, features_names_y, features_names_t))
    features_names = [nm for nms in features_names for nm in nms]  # flatten list

    return pd.Series(features, index=features_names)


def dms_to_dd(d, m, s):
    """Coordinate degrees, minutes, seconds
    to decimal degrees"""
    return d + m / 60 + s / 3600


def latlon_dms_to_dd(coords):
    """Convert latitude and longitude in
    degrees minutes seconds to decimal degrees.

    Accepts coords as a string
    in the form DDMMSS(N/S)DDDMMSS(E/W)"""
    # Example 570613N0095944E
    lat_d, lat_m, lat_s = float(coords[:2]), float(coords[2:4]), float(coords[4:6])
    lat_dir = coords[6]
    lon_d, lon_m, lon_s = float(coords[7:10]), float(coords[10:12]), float(coords[12:14])
    lon_dir = coords[14]

    lat_dd = dms_to_dd(lat_d, lat_m, lat_s)
    lon_dd = dms_to_dd(lon_d, lon_m, lon_s)

    lat_dd = -lat_dd if lat_dir == 'S' else lat_dd
    lon_dd = -lon_dd if lon_dir == 'W' else lon_dd

    return lat_dd, lon_dd


for data_file in data_files:
    data = pd.read_pickle(path_to_data + '/' + data_file)
    data.dropna(how='any', inplace=True)
    print "init columns", data.columns

    start_date, end_date = data_file.split('_')[2], data_file.split('_')[3][:-4]

    # extract unique flights ids
    flights = data.copy()
    flights.drop_duplicates(['IFPLID', 'TARGET_DATE', 'AIRCRAFT_TYPE', 'CALLSIGN', 'ADEP', 'ADES'], inplace=True)
    flights.dropna(how='any', inplace=True)
    flights.drop(['NCP', 'XCP', 'SNAPSHOT_TIME', 'LAT', 'LON', 'ALT', 'TIME_'], inplace=True, axis=1)
    flights.reset_index(inplace=True, drop=True)
    print "Number of flights:", flights.shape[0]

    # drop flights where the time interval between the first
    # observation and the entry is less than 5 minutes
    flights['TO_BORDER'] = (pd.to_datetime(flights['ENTRY_T']) - pd.to_datetime(flights['CORR_T'])).dt.total_seconds()
    flights = flights[flights['TO_BORDER'] >= 5 * 60]
    flights.reset_index(inplace=True, drop=True)
    print "interim columns", flights.columns
    print "Number of flights after min distance dropped:", flights.shape[0]

    if N_OBSERV == 1:
        # if we need only one observation
        # convert the correlation lat and lon (i.e. first observation)
        # to x and y
        flights[['X1', 'Y1']] = flights.apply(lambda row: pd.Series(conv_coord.lalo_to_xy(row['CORR_LAT'], row['CORR_LON'])),
                                              axis=1)
    else:
        # add predictors X,Y,T from the flown trajectory
        # according to the required number of observations
        # this process takes a very long time
        flights = flights.join(flights.apply(lambda row: prepare_trajectory(data[(data['IFPLID'] == row['IFPLID']) & (data['TARGET_DATE'] == row['TARGET_DATE'])]),
                                             axis=1))

    # drop flights where Y1 is positive, as those flights
    # are not coming from ROU
    flights = flights[flights['Y1'] < 0]
    flights.reset_index(inplace=True, drop=True)
    print "Number of flights after wrong Y1 dropped:", flights.shape[0]

    # convert entry time to seconds elapsed since midnight
    flights['ENTRY_T_SEC'] = (pd.to_datetime(flights['ENTRY_T']) - pd.to_datetime(flights['TARGET_DATE'])).dt.total_seconds()

    # convert ncop and xcop lat/lon to XY
    coord_points = pd.read_csv(resources_folder + '/adapt-points', sep=';', header=None)
    coord_points.rename(columns={0: 'pid', 9: 'latlon'}, inplace=True)
    coord_points['pid'] = coord_points.apply(lambda row: row['pid'].strip(), axis=1)
    coord_points.drop_duplicates(['pid'], inplace=True)
    coord_points[['lat', 'lon']] = coord_points.apply(lambda row: pd.Series(latlon_dms_to_dd(row['latlon'])), axis=1)
    coord_points[['x', 'y']] = coord_points.apply(lambda row: pd.Series(conv_coord.lalo_to_xy(row['lat'], row['lon'])),
                                                  axis=1)

    flights = pd.merge(flights, coord_points[['pid', 'x', 'y', 'lat', 'lon']], how='left', left_on='NCOP', right_on='pid')
    flights.rename(columns={'x': 'NCOP_X', 'y': 'NCOP_Y', 'lat': 'NCOP_LAT', 'lon': 'NCOP_LON'}, inplace=True)
    flights.drop('pid', axis=1, inplace=True)
    flights.dropna(subset=['NCOP_X', 'NCOP_Y', 'NCOP_LAT', 'NCOP_LON'], inplace=True)
    flights = pd.merge(flights, coord_points[['pid', 'x', 'y', 'lat', 'lon']], how='left', left_on='XCOP', right_on='pid')
    flights.rename(columns={'x': 'XCOP_X', 'y': 'XCOP_Y', 'lat': 'XCOP_LAT', 'lon': 'XCOP_LON'}, inplace=True)
    flights.drop('pid', axis=1, inplace=True)
    flights.dropna(subset=['XCOP_X', 'XCOP_Y', 'XCOP_LAT', 'XCOP_LON'], inplace=True)

    # add wake turbulence category based on aircraft type
    wtc = pd.read_csv(resources_folder + '/aircraft-type-wtc', sep=';')
    wtc.dropna(how='any', inplace=True)
    wtc['a_type'] = wtc.apply(lambda row: row['a_type'].strip(), axis=1)
    wtc.drop_duplicates(['a_type'], inplace=True)
    flights = pd.merge(flights, wtc[['a_type', 'WTC']], how='left', left_on='AIRCRAFT_TYPE', right_on='a_type')
    flights.drop('a_type', axis=1, inplace=True)

    # add encoded day of week (DOW) as sin and cos of DOW
    flights['DOW'] = flights.apply(lambda row: pd.Series(row['TARGET_DATE']).dt.dayofweek, axis=1)
    flights = encode_cyclic(flights, 'DOW', 7)

    # add encoded hour since midnight (HOUR) of the first observation as sin and cos of HOUR
    # flights['HOUR'] = flights.apply(lambda row: pd.Series(row['T1']).dt.hour, axis=1)
    flights['HOUR'] = flights.apply(lambda row: pd.Series(row['CORR_T']).dt.hour, axis=1)
    flights = encode_cyclic(flights, 'HOUR', 24)

    print "final columns", flights.columns

    # before splitting the data sample, pickle it
    flights.to_pickle(interim_data_folder + '/interim_data_sample_%s_%s.pkl' % (start_date, end_date))

