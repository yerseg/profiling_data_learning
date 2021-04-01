import os

import pandas as pd
import numpy as np
from datetime import datetime as dt
import scipy.stats as stats
from geopy.distance import distance as geodist

BASE_BT_NAME = 'base_bt'
BASE_WIFI_NAME = 'base_wifi'
BROADCASTS_NAME = 'broadcasts'
CONN_WIFI_NAME = 'conn_wifi'
LE_BT_NAME = 'le_bt'
LOCATION_NAME = 'location'

TIME_SAMPLE_FREQ = '30s'


def location_generate_features(file, out_path):
    print(file)

    df = pd.read_csv(file, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'accuracy', 'altitude', 'latitude', 'longitude', 'user'])

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    VALID_USER = df.iloc[0]['user']
    df['events_count'] = 1

    df['accuracy'] = df['accuracy'].apply(lambda x: x.replace(',', '.'))
    df['altitude'] = df['altitude'].apply(lambda x: x.replace(',', '.'))
    df['latitude'] = df['latitude'].apply(lambda x: x.replace(',', '.'))
    df['longitude'] = df['longitude'].apply(lambda x: x.replace(',', '.'))

    df['accuracy'] = df['accuracy'].astype(float)
    df['altitude'] = df['altitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    df['prev_latitude'] = df['latitude'].shift(1)
    df['prev_longitude'] = df['longitude'].shift(1)
    df['prev_timestamp'] = df['timestamp'].shift(1)
    df['prev_altitude'] = df['altitude'].shift(1)

    def get_speed(row):
        prev_coords = (row['prev_latitude'], row['prev_longitude'])
        curr_coords = (row['latitude'], row['longitude'])
        delta = row['timestamp'] - row['prev_timestamp']
        if pd.isnull(delta):
            return np.nan
        time = abs(delta.total_seconds())
        if np.isnan(prev_coords[0]) or np.isnan(prev_coords[1]) or np.isnan(curr_coords[0]) or np.isnan(curr_coords[1]):
            return np.nan
        if time == 0:
            return np.nan
        return geodist(curr_coords, prev_coords).meters / time

    def get_altitude_speed(row):
        prev = row['prev_altitude']
        curr = row['altitude']
        delta = row['timestamp'] - row['prev_timestamp']
        if pd.isnull(delta):
            return np.nan
        time = abs(delta.total_seconds())
        if np.isnan(prev) or np.isnan(curr):
            return np.nan
        if time == 0:
            return np.nan
        return abs(curr - prev) / time

    df['speed'] = df.apply(lambda row: get_speed(row), axis=1)
    df['altitude_speed'] = df.apply(lambda row: get_altitude_speed(row), axis=1)

    df = df.drop(['prev_latitude', 'prev_longitude', 'prev_altitude'], axis=1)

    df['prev_speed'] = df['speed'].shift(1)
    df['prev_altitude_speed'] = df['altitude_speed'].shift(1)

    df = df.drop(['prev_altitude_speed', 'prev_speed', 'timestamp', 'prev_timestamp'], axis=1)

    def kurt(col):
        return stats.kurtosis(col)

    def user_agg(col):
        if (col == VALID_USER).all():
            return 1
        else:
            return 0

    common_funcs_list = ['mean', 'var', 'median', 'skew', kurt, 'std']

    agg_dict = {
        'accuracy': common_funcs_list,
        'speed': common_funcs_list,
        'altitude_speed': common_funcs_list,
        'events_count': 'sum',
        'user': user_agg
    }

    df_sampling = df.groupby(pd.Grouper(freq=TIME_SAMPLE_FREQ)).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling = df.rolling(TIME_SAMPLE_FREQ, min_periods=1, center=False).agg(agg_dict)

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    df_sampling.to_csv(os.path.join(out_path, "location_sampling_ds_" + str(file[-6:])))
    df_rolling.to_csv(os.path.join(out_path, "location_rolling_ds_" + str(file[-6:])))


def main():
    base_path = ".\\_events\\_generated"
    for user_dir in os.listdir(base_path):
        u_dir = os.path.join(base_path, user_dir)
        count = 0
        for f in os.listdir(u_dir):
            if int(f[-6]) > count:
                count = int(f[-6])

        for i in range(count):
            new_path = os.path.join(u_dir, TIME_SAMPLE_FREQ)
            location_generate_features(os.path.join(u_dir, LOCATION_NAME + '_' + str(i) + ".data"), new_path)


if __name__ == '__main__':
    main()