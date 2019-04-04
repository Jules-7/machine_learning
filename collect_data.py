""" Collect Data For Machine Learning

This module collects the data required for Machine Learning.

The data contains:
- flight identification information
- flown trajectory
- entry conditions

The data is extracted from two tables:
    - ATM.FLIGHT: contains flown trajectories, ifplid, target_date,
                callsign, aircraft type.
    - ATM.IFMP_FLIGHT: contains ifplid, target_date,
                callsign, aircraft type, ncop, xcop.

AOR entries are computed by overlapping the flown trajectory
and AOR boundary geometry.

Filters:
    - Flights coming not from ROU center are discarded.
    - Flights below FL245 are discarded.
    - Flights appearing within AOR (i.e. the first observation is
    within AOR) are discarded.
    - Flights with AOR entry time before 01:00 and after 23;00
    are discarded.

NOTE: while AOR entries are filtered to be in the range
    from 01:00 on target_date to 23:00 on target_date,
    the points in the flown trajectory are not filtered
    and might have a timestamp of the previous or the next day

Created: 14/03/19

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from mpl_toolkits.basemap import Basemap
from shapely.geometry.polygon import LinearRing, LineString
from shapely.geometry import asLineString, asPoint, Point, MultiPoint, GeometryCollection, Polygon
from CoorConv import CoorConv
from connect_to_db import db_connection


# ###################### GLOBAL SETTINGS ############################
path_to_project = os.path.dirname(os.path.dirname(__file__))

EARTH_RADIUS = 6371e3  # [m], mean radius

N_OBSERV = 15  # number of 'observed' points from trajectory --> features
PERIOD = 5  # interval between 'observed' points

# class for coordinates conversion
conv_coord = CoorConv(51, 8)  # MUAC airspace center: 51 deg lat, 8 deg long

resources_folder = 'resources'
data_folder = 'data'

# convert AOR to shapely object
aor = pd.read_csv('resources/aor')
aor_coords = list(zip(aor['lat'].values.tolist(), aor['lon'].values.tolist()))
# LinearRing is a closed geometry: first
aor_geom = LinearRing([conv_coord.lalo_to_xy(lat, lon) for (lat, lon) in aor_coords])
aor_polygon = Polygon([conv_coord.lalo_to_xy(lat, lon) for (lat, lon) in aor_coords])
# ################### END GLOBAL CONSTANTS ##########################


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def dms_to_dd(d, m, s):
    """Coordinate degrees, minutes, seconds
    to decimal degrees"""
    return d + m/60 + s/3600


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


def correct_ifplid(ifplid):
    if len(ifplid) != 10:
        if len(ifplid) != 8:
            ifplid = ifplid.rjust(8, '0')
        ifplid = 'AA' + ifplid
    return ifplid


def flight_entry(trajectory):

    # interpolate trajectory
    trajectory_interp = interpolate_trajectory(trajectory)
    trajectory_interp.reset_index(inplace=True, drop=True)

    # check whether the first point of a trajectory is within AOR
    # if that is the case, return None
    first_point = asPoint(trajectory_interp.loc[0, ['X', 'Y']].values)
    if first_point.within(aor_polygon):
        # print("trajectory started within AOR")
        return pd.Series([None, None, None], index=['ENTRY_X', 'ENTRY_Y', 'ENTRY_T'])

    # convert trajectory to shapely object
    trajectory_geom = asLineString(trajectory_interp.loc[:, ['X', 'Y']].values)

    # intersect trajectory with AOR
    intersection = trajectory_geom.intersection(aor_geom)
    inters_geom = intersection.geom_type

    if not isinstance(intersection, Point) and not isinstance(intersection, MultiPoint) and not isinstance(intersection, GeometryCollection):
        print bcolors.WARNING + "unknown geometry type" + bcolors.ENDC, inters_geom, '\n'

    if isinstance(intersection, Point):
        # trajectory intersects AOR at one point - entry
        entry_x, entry_y = intersection.x, intersection.y

    elif isinstance(intersection, MultiPoint):
        # trajectory intersects AOR at two points - entry and exit
        # since Shapely intersection method does not return
        # intersection points in the right order (i.e. first
        # intersection close to the start of the trajectory),
        # compute the distance from the first observation
        # to each point and take the one with the smallest distance
        distances = []
        for point in intersection.geoms:
            distances.append(np.sqrt((point.x - trajectory_interp.loc[0, 'X'])**2 + (point.y - trajectory_interp.loc[0, 'Y'])**2))

        entry_point = intersection.geoms[np.argmin(distances)]
        entry_x, entry_y = entry_point.x, entry_point.y

    elif isinstance(intersection, GeometryCollection) and intersection.is_empty:
        # if there is no intersection - there is no entry point
        entry_x, entry_y = None, None

    if entry_x and entry_y:
        # estimate the time of intersection as the time at a point on-route
        # closest to the entry point (i.e. smallest distance)
        trajectory_interp['DIST'] = trajectory_interp.apply(lambda row: np.sqrt((row['X'] - entry_x)**2 + (row['Y'] - entry_y)**2), axis=1)
        entry_t = trajectory_interp.sort_values(by='DIST').iloc[0, :]['TIME_']
    else:
        entry_t = None

    return pd.Series([entry_x, entry_y, entry_t], index=['ENTRY_X', 'ENTRY_Y', 'ENTRY_T'])


conn = db_connection()  # connect to database

df_trajectories = pd.DataFrame()

# extract data for the dates below
start_date, end_date = datetime.date(2017, 1, 22), datetime.date(2019, 2, 22)
no_data_days = ['2017-02-03', '2017-02-22', '2017-04-05', '2017-04-23', '2017-07-04', '2017-08-31', '2017-09-09',
                '2017-10-20', '2017-10-27', '2017-11-09', '2018-01-01', '2018-01-02', '2018-01-12', '2018-02-17',
                '2018-03-22', '2018-04-09', '2018-04-30', '2018-05-09', '2018-05-27', '2018-06-22', '2018-08-21',
                '2018-10-23', '2018-11-09', '2018-11-18', '2018-12-02', '2019-01-08', '2019-01-22']
delta = end_date - start_date
numdays = delta.days
target_dates = [(start_date + datetime.timedelta(days=x)).isoformat() for x in range(0, numdays)]

start_run_time = datetime.datetime.now()

for target_date in target_dates:

    if target_date in no_data_days:
        continue

    print "\n", target_date

    # load flown trajectories
    trajectories = pd.read_sql("""select ifplid, target_date, aircraft_type, callsign, adep, ades,
                                    t.x lon, t.y lat, t.z*100 alt, 
                                    TO_DATE('19700101','yyyymmdd') + (t.w/1000/24/60/60) time_
                                 from atm.flight f, table(sdo_util.getvertices(f.trajectory)) t
                                 where target_date = to_date('%s', 'YYYY-MM-DD')
                                 """ % target_date, conn)
    trajectories.dropna(how='any', inplace=True)
    trajectories['IFPLID'] = trajectories['IFPLID'].apply(lambda id_val: correct_ifplid(id_val))

    # extract the first observation (visible to radar and when ABI is received) of each flights
    first_observ = trajectories.sort_values(by='TIME_').groupby('IFPLID').first()
    first_observ.reset_index(level=0, inplace=True)
    first_observ.rename(columns={'LAT': 'CORR_LAT', 'LON': 'CORR_LON', 'ALT': 'CORR_ALT', 'TIME_': 'CORR_T'}, inplace=True)

    # filter lateral entries
    lateral_entry = first_observ[first_observ['CORR_ALT'] >= 24500 * 0.3048]

    # cannot use this value since it wont be available in real-time system
    # extract last observations (cancel assume -> not under control anymore)
    # last_observ = trajectories.sort_values(by='T').groupby('IFPLID').last()
    # last_observ.reset_index(level=0, inplace=True)
    # last_observ.rename(columns={'LAT': 'LAST_LAT', 'LON': 'LAST_LON', 'ALT': 'LAST_ALT', 'T': 'LAST_T'}, inplace=True)

    # load flight related information from the IFMP
    flights_info = pd.read_sql("""select ifpl_id, target_date, aircraft_id, aircraft_type, adep, ades, 
                                        ncop, ncp, xcop, xcp, snapshot_time
                                    from atm.ifmp_flight 
                                    where target_date = to_date('%s', 'YYYY-MM-DD')
                                    """ % target_date, conn)
    flights_info = flights_info.sort_values(by='SNAPSHOT_TIME').groupby(['IFPL_ID', 'TARGET_DATE', 'AIRCRAFT_ID', 'AIRCRAFT_TYPE', 'ADEP', 'ADES']).last().reset_index()
    flights_info.dropna(how='any', inplace=True)
    flights_info.rename(columns={'IFPL_ID': 'IFPLID', 'AIRCRAFT_ID': 'CALLSIGN'}, inplace=True)

    # select flights coming only from ROU upstream sector
    flights_info = flights_info[flights_info['NCP'] == 'ROU']

    # flights coming from ROU and are at an altitude > FL245
    flights = pd.merge(lateral_entry, flights_info, how='inner', on=['IFPLID', 'TARGET_DATE', 'CALLSIGN', 'AIRCRAFT_TYPE', 'ADES', 'ADEP'])

    # compute entry point by intersecting the flown trajectory and AOR
    flights = flights.join(flights.apply(lambda row: flight_entry(trajectories[(trajectories['IFPLID'] == row['IFPLID'])]), axis=1), rsuffix='a')

    # drop flights whose entries are None
    flights.dropna(subset=['ENTRY_X', 'ENTRY_Y', 'ENTRY_T'], inplace=True)

    # limit flights to those that entered AOR on the target_date from 01:00 to 23:00
    min_time = pd.to_datetime(target_date) + pd.DateOffset(hours=1)
    max_time = pd.to_datetime(target_date) + pd.DateOffset(hours=23)
    flights = flights[(flights['ENTRY_T'] >= min_time) & (flights['ENTRY_T'] <= max_time)]
    flights.reset_index(inplace=True, drop=True)

    trajectories_selected = pd.merge(flights, trajectories, how='inner', on=['IFPLID', 'TARGET_DATE', 'CALLSIGN', 'AIRCRAFT_TYPE', 'ADES', 'ADEP'])

    if flights['IFPLID'].nunique() != trajectories_selected['IFPLID'].nunique():
        print "different number of unique flights flights"

    df_trajectories = df_trajectories.append(trajectories_selected, ignore_index=True)

    print "day proccessed in ", datetime.datetime.now() - start_run_time
    start_run_time = datetime.datetime.now()

    # # discard flights where the number of observations is less than min_obs
    # # flights_n_obs = flights.groupby(['IFPLID', 'TARGET_DATE']).size().reset_index().rename(columns={0: 'COUNT'})
    # # discard_flights = flights_n_obs[flights_n_obs['COUNT'] < 12]

conn.close()  # all data is read

print 'size of the final dataframe', df_trajectories.shape[0]
df_trajectories.to_pickle(data_folder + '/data_sample_%s_%s.pkl' % (target_dates[0], target_dates[-1]))
