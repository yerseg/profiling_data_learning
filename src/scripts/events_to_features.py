import os

import pandas as pd
import numpy as np
from datetime import datetime as dt
import scipy.stats as stats
from geopy.distance import distance as geodist
from scipy.spatial import distance


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

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)]) for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling = df.rolling(TIME_SAMPLE_FREQ, min_periods=1, center=False).agg(agg_dict)

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)]) for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    df_sampling.to_csv(os.path.join(out_path, "location_sampling_ds_" + str(file[-6:])))
    df_rolling.to_csv(os.path.join(out_path, "location_rolling_ds_" + str(file[-6:])))


def wifi_generate_features(file, conn_file, out_path):
    print(file, conn_file)
    df = pd.read_csv(file, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'uuid', 'bssid', 'chwidth', 'freq', 'level', 'user'])

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    VALID_USER = df.iloc[0]['user']
    df['events_count'] = 1

    df = df.drop(['timestamp', 'chwidth'], axis=1)
    bssid_map = {bssid.replace(' ', ''): idx for bssid, idx in zip(df.bssid.unique(), range(len(df.bssid.unique())))}

    df.bssid = df.bssid.apply(lambda x: str(x).replace(' ', ''))
    df.level = df.level.apply(lambda x: str(x).replace(' ', ''))
    df.freq = df.freq.apply(lambda x: str(x).replace(' ', ''))

    df['bssid_level'] = df[['bssid', 'level']].agg(','.join, axis=1)
    df['count'] = 1

    def user_agg(col):
        if (col == VALID_USER).all():
            return 1
        else:
            return 0

    def agg_string_join(col):
        col = col.apply(lambda x: str(x))
        return col.str.cat(sep=',').replace(' ', '')

    def agg_bssid_col(col):
        array_len = len(bssid_map)
        array = np.zeros(array_len, dtype='float')

        def fill_array(x):
            tmp = x.split(',')
            bssid = tmp[0]
            level = float(tmp[1])
            array[bssid_map[bssid.replace(' ', '')]] = level
            return

        col.apply(lambda x: fill_array(x))
        return np.array2string(array, separator=',').replace(' ', '')[1:-1]

    all_func_dicts_quantum = {'freq': agg_string_join, 'level': agg_string_join, 'bssid_level': agg_bssid_col,
                              'count': 'sum',
                              'events_count': 'sum', 'user': user_agg}

    df_quantum = df.groupby(['timestamp', 'uuid'], as_index=True).agg(all_func_dicts_quantum)

    df_quantum = df_quantum.reset_index()
    df_quantum.index = pd.DatetimeIndex(df_quantum.timestamp)

    df_quantum = df_quantum[df_quantum['count'] != 0]

    df_conn = pd.read_csv(conn_file, sep=';', index_col=False,
                          header=None, low_memory=False, names=['timestamp', '1', 'bssid', '2', '3', '4', '5', 'level', '6'])

    df_conn = df_conn.drop(df_conn.columns.difference(['bssid', 'timestamp', 'level']), axis=1)
    df_conn.index = pd.DatetimeIndex(df_conn.timestamp)
    df_conn = df_conn.sort_index()

    def get_level_from_row(row):
        bssid = df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['bssid']
        if str(bssid) == 'nan' or str(bssid) == 'null' or str(bssid) == '':
            return 0

        level = df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['level']
        time = dt.strptime(df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['timestamp'],
                           '%Y-%m-%d %H:%M:%S.%f')
        return level if abs((time - row.name).total_seconds()) <= 10 else 0

    # df_quantum['conn_level'] = df_quantum.apply(lambda row: get_level_from_row(row), axis = 1)

    def string2array(string):
        try:
            array = np.fromstring(string, sep=',')
            return array
        except:
            return np.nan

    def to_ones_array(array):
        try:
            array[array != 0] = 1
            return array
        except:
            return np.nan

    def get_len(obj):
        try:
            length = len(obj)
            return length
        except:
            return np.nan

    def get_occured_nets_count(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        intersection = np.logical_and(curr, prev)
        diff = np.logical_and(curr, np.logical_not(intersection))

        if (np.count_nonzero(np.logical_or(prev, curr)) == 0):
            return 0

        return np.count_nonzero(diff) / np.count_nonzero(np.logical_or(prev, curr))

    def get_disappeared_nets_count(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        intersection = np.logical_and(curr, prev)
        diff = np.logical_and(prev, np.logical_not(intersection))

        if (np.count_nonzero(np.logical_or(prev, curr)) == 0):
            return 0

        return np.count_nonzero(diff) / np.count_nonzero(np.logical_or(prev, curr))

    def get_jaccard_index(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        return distance.jaccard(prev, curr)

    def get_occur_speed(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        return np.linalg.norm(prev - curr) / np.sqrt(get_len(prev))

    def get_level_speed(row, prev_col, curr_col):
        prev = string2array(row[prev_col])
        curr = string2array(row[curr_col])
        return np.linalg.norm(prev - curr) / np.sqrt(get_len(prev))

    def calc_single_cols_in_window(df, col, new_col, window, func):
        def func_wrapper(func, row, prev_col, curr_col):
            delta = row.timestamp - row.prev_timestamp
            if pd.isnull(delta):
                delta = 0
            else:
                delta = abs(delta.total_seconds())
            if delta > 10 * 60:
                return np.nan
            else:
                return func(row, prev_col_name, col)

        new_cols = []

        for i in range(window):
            prev_col_name = "_".join(['prev', col, str(i + 1)])
            new_col_name = "_".join([new_col, str(i + 1)])

            df['prev_timestamp'] = df.timestamp.shift(i + 1)
            df[prev_col_name] = df[col].shift(i + 1)
            df[new_col_name] = df.apply(lambda row: func_wrapper(func, row, prev_col_name, col), axis=1)
            df = df.drop(prev_col_name, axis=1)
            df = df.drop('prev_timestamp', axis=1)
            new_cols.append(new_col_name)

        df["_".join([new_col, 'mean'])] = df[new_cols].mean(axis=1)
        df["_".join([new_col, 'median'])] = df[new_cols].median(axis=1)
        df["_".join([new_col, 'var'])] = df[new_cols].var(axis=1)

        return df

    WINDOW_SIZE = 5

    occur_and_level_columns_map = [
        ("bssid_level", "occured_nets_count", WINDOW_SIZE, get_occured_nets_count),
        ("bssid_level", "disappeared_nets_count", WINDOW_SIZE, get_disappeared_nets_count),
        ("bssid_level", "jaccard_index", WINDOW_SIZE, get_jaccard_index),
        ("bssid_level", "occur_speed", WINDOW_SIZE, get_occur_speed),
        ("bssid_level", "level_speed", WINDOW_SIZE, get_level_speed)
    ]

    for (col, new_col, window, func) in occur_and_level_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, window, func)

    def get_conn_level_speed(row, prev_col, curr_col):
        return row[curr_col] - row[prev_col]

    single_columns_map = [
        #     ("conn_level", "conn_level_speed", WINDOW_SIZE, get_conn_level_speed),
        ("count", "count_speed", WINDOW_SIZE, get_conn_level_speed)
    ]

    for (col, new_col, window, func) in single_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, window, func)

    def agg_str(col):
        #     all_freq = col.str.cat(sep=',')
        return string2array(col)

    def str_mean(col):
        array = agg_str(col)
        if str(array) == 'nan':
            return 0
        return np.mean(array)

    def mean(col):
        return np.mean(col)

    def var(col):
        return np.var(col)

    def median(col):
        return np.median(col)

    def skew(col):
        return stats.skew(col)

    def kurt(col):
        return stats.kurtosis(col)

    def user_agg(col):
        if (col == 1).all():
            return 1
        else:
            return 0

    df_quantum['freq'] = df_quantum.apply(lambda row: str_mean(row['freq']), axis=1)
    df_quantum['level'] = df_quantum.apply(lambda row: str_mean(row['level']), axis=1)

    cols_for_drop = []
    names = [
        "occured_nets_count",
        "disappeared_nets_count",
        "jaccard_index",
        "occur_speed",
        "count_speed",
        #     "conn_level_speed",
        "level_speed",
        "count_speed"
    ]

    for i in range(1, WINDOW_SIZE + 1):
        for name in names:
            cols_for_drop.append('_'.join([name, str(i)]))

    df_quantum = df_quantum.drop(['bssid_level', 'timestamp', 'uuid'], axis=1)
    df_quantum = df_quantum.drop(cols_for_drop, axis=1)

    df_quantum.columns

    common_cols = [x for x in df_quantum.columns[0:5] if x != 'user' and x != 'events_count']
    speed_acc_cols = df_quantum.columns[5:]

    common_funcs_list = [mean, var, median, skew, kurt]
    special_funcs_list = [mean, pd.DataFrame.mad, skew]

    common_cols_map = {col: common_funcs_list for col in common_cols}
    speed_acc_cols_map = {col: special_funcs_list for col in speed_acc_cols}

    additional = {'user': user_agg, 'events_count': 'sum'}

    agg_dict = common_cols_map
    agg_dict.update(speed_acc_cols_map)
    agg_dict.update(additional)

    df_quantum[speed_acc_cols] = df_quantum[speed_acc_cols].apply(pd.to_numeric)

    df_sampling = df_quantum.groupby(pd.Grouper(freq=TIME_SAMPLE_FREQ)).agg(agg_dict)

    df_rolling = df_quantum.rolling(TIME_SAMPLE_FREQ, min_periods=1, center=False).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    df_sampling.to_csv(os.path.join(out_path, "wifi_sampling_ds_" + str(file[-6:])))
    df_rolling.to_csv(os.path.join(out_path, "wifi_rolling_ds_" + str(file[-6:])))


def bt_generate_features(file, le_file, out_path):
    print(file, le_file)
    df = pd.read_csv(file, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'action', 'bssid', 'major_class', 'class', \
                            'bond_state', 'type', 'user'])

    df = df[df['action'] == 'android.bluetooth.device.action.FOUND']

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    VALID_USER = df.iloc[0]['user']
    df['events_count'] = 1

    df = df.drop(['timestamp', 'action', 'class', 'major_class', 'bond_state', 'type'], axis=1)

    bssid_map = {bssid.replace(' ', ''): idx for bssid, idx in zip(df.bssid.unique(), range(len(df.bssid.unique())))}
    df.bssid = df.bssid.apply(lambda x: str(x).replace(' ', ''))

    df['count'] = 1

    def agg_string_join(col):
        col = col.apply(lambda x: str(x))
        return col.str.cat(sep=',').replace(' ', '')

    def agg_bssid_col(col):
        array_len = len(bssid_map)
        array = np.zeros(array_len, dtype='int8')

        def fill_array(bssid):
            array[bssid_map[bssid.replace(' ', '')]] = 1
            return

        col.apply(lambda x: fill_array(x))
        return np.array2string(array, separator=',').replace(' ', '')[1:-1]

    one_hot_columns_count = 0
    for col in df.columns:
        if col.find('one_hot') != -1:
            one_hot_columns_count += 1

    def user_agg(col):
        if (col == VALID_USER).all():
            return 1
        else:
            return 0

    cat_columns = df.columns[1:1 + one_hot_columns_count]
    cat_columns_map = {col: 'mean' for col in cat_columns}

    all_func_dicts_quantum = {'bssid': agg_bssid_col, 'count': 'sum', 'user': user_agg, 'events_count': 'sum'}
    all_func_dicts_quantum.update(cat_columns_map)

    df_quantum = df.groupby(pd.Grouper(freq='5s'), as_index=True).agg(all_func_dicts_quantum)

    df_quantum = df_quantum.reset_index()
    df_quantum.index = pd.DatetimeIndex(df_quantum.timestamp)

    df_quantum = df_quantum.dropna()

    df_le = pd.read_csv(le_file, sep=';', index_col=False, header=None,
                        low_memory=False, names=['timestamp', '1', '2', '3', 'level', '3', 'connectable', '4'])

    df_le = df_le.drop(df_le.columns.difference(['connectable', 'timestamp', 'level']), axis=1)
    df_le.index = pd.DatetimeIndex(df_le.timestamp)
    df_le = df_le.sort_index()

    df_le['connectable'] = df_le['connectable'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

    df_le = df_le.groupby(pd.Grouper(freq='5s'), as_index=True).agg({'level': 'mean', 'connectable': 'mean'})

    df_le = df_le.dropna()

    def get_le_conn_status_from_row(row):
        conn = df_le.iloc[df_le.index.get_loc(row.name, method='nearest')]['connectable']
        time = df_le.iloc[df_le.index.get_loc(row.name, method='nearest')].name
        return conn if abs((time - row.name).total_seconds()) < 10 else 0

    def get_le_level_from_row(row):
        level = df_le.iloc[df_le.index.get_loc(row.name, method='nearest')]['level']
        time = df_le.iloc[df_le.index.get_loc(row.name, method='nearest')].name
        return level if abs((time - row.name).total_seconds()) < 10 else 0

    df_quantum['le_connectable'] = df_quantum.apply(lambda row: get_le_conn_status_from_row(row), axis=1)
    df_quantum['le_level'] = df_quantum.apply(lambda row: get_le_level_from_row(row), axis=1)

    def string2array(string):
        try:
            array = np.fromstring(string, sep=',')
            return array
        except:
            return np.nan

    def to_ones_array(array):
        try:
            array[array != 0] = 1
            return array
        except:
            return np.nan

    def get_len(obj):
        try:
            length = len(obj)
            return length
        except:
            return np.nan

    def get_occured_nets_count(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        intersection = np.logical_and(curr, prev)
        diff = np.logical_and(curr, np.logical_not(intersection))

        if (np.count_nonzero(np.logical_or(prev, curr)) == 0):
            return 0

        return np.count_nonzero(diff) / np.count_nonzero(np.logical_or(prev, curr))

    def get_disappeared_nets_count(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        intersection = np.logical_and(curr, prev)
        diff = np.logical_and(prev, np.logical_not(intersection))

        if (np.count_nonzero(np.logical_or(prev, curr)) == 0):
            return 0

        return np.count_nonzero(diff) / np.count_nonzero(np.logical_or(prev, curr))

    def get_jaccard_index(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        return distance.jaccard(prev, curr)

    def get_occur_speed(row, prev_col, curr_col):
        prev = to_ones_array(string2array(row[prev_col]))
        curr = to_ones_array(string2array(row[curr_col]))
        return np.linalg.norm(prev - curr) / np.sqrt(get_len(prev))

    def calc_single_cols_in_window(df, col, new_col, window, func):
        def func_wrapper(func, row, prev_col, curr_col):
            delta = row.timestamp - row.prev_timestamp
            if pd.isnull(delta):
                delta = 0
            else:
                delta = abs(delta.total_seconds())
            if delta > 10 * 60:
                return np.nan
            else:
                return func(row, prev_col_name, col)

        new_cols = []

        for i in range(window):
            prev_col_name = "_".join(['prev', col, str(i + 1)])
            new_col_name = "_".join([new_col, str(i + 1)])

            df.loc[:, 'prev_timestamp'] = df.timestamp.shift(i + 1)
            df.loc[:, prev_col_name] = df[col].shift(i + 1)
            df.loc[:, new_col_name] = df.apply(lambda row: func_wrapper(func, row, prev_col_name, col), axis=1)
            df = df.drop(prev_col_name, axis=1)
            df = df.drop('prev_timestamp', axis=1)
            new_cols.append(new_col_name)

        df.loc[:, "_".join([new_col, 'mean'])] = df[new_cols].mean(axis=1)
        df.loc[:, "_".join([new_col, 'median'])] = df[new_cols].median(axis=1)
        df.loc[:, "_".join([new_col, 'var'])] = df[new_cols].var(axis=1)

        return df

    WINDOW_SIZE = 5

    occur_and_level_columns_map = [
        ("bssid", "occured_devices_count", WINDOW_SIZE, get_occured_nets_count),
        ("bssid", "disappeared_devices_count", WINDOW_SIZE, get_disappeared_nets_count),
        ("bssid", "jaccard_index", WINDOW_SIZE, get_jaccard_index),
        ("bssid", "occur_speed", WINDOW_SIZE, get_occur_speed)
    ]

    for (col, new_col, window, func) in occur_and_level_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, window, func)

    def get_conn_level_speed(row, prev_col, curr_col):
        return row[curr_col] - row[prev_col]

    single_columns_map = [
        ("count", "count_speed", WINDOW_SIZE, get_conn_level_speed)
    ]

    for (col, new_col, window, func) in single_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, window, func)

    def agg_str(col):
        all_freq = col.str.cat(sep=',')
        return string2array(all_freq)

    def str_mean(col):
        return np.mean(agg_str(col))

    def str_var(col):
        return np.var(agg_str(col))

    def str_median(col):
        return np.median(agg_str(col))

    def str_skew(col):
        return stats.skew(agg_str(col))

    def str_kurt(col):
        return stats.kurtosis(agg_str(col))

    def mean(col):
        return np.mean(col)

    def var(col):
        return np.var(col)

    def median(col):
        return np.median(col)

    def skew(col):
        return stats.skew(col)

    def kurt(col):
        return stats.kurtosis(col)

    cols_for_drop = []
    names = [
        "occured_devices_count",
        "disappeared_devices_count",
        "jaccard_index",
        "occur_speed",
        "count_speed"
    ]

    for i in range(1, WINDOW_SIZE + 1):
        for name in names:
            cols_for_drop.append('_'.join([name, str(i)]))

    df_quantum = df_quantum.drop(['bssid', 'timestamp'], axis=1)
    df_quantum = df_quantum.drop(cols_for_drop, axis=1)

    def user_agg(col):
        if (col == 1).all():
            return 1
        else:
            return 0

    common_cols = [x for x in df_quantum.columns[:one_hot_columns_count + 3] if x != 'user' and x != 'events_count']
    speed_acc_cols = df_quantum.columns[one_hot_columns_count + 3:]

    common_funcs_list = [mean, var, median, skew, kurt]
    special_funcs_list = [mean, pd.DataFrame.mad, skew]

    common_cols_map = {col: common_funcs_list for col in common_cols}
    speed_acc_cols_map = {col: special_funcs_list for col in speed_acc_cols}

    additional = {'user': user_agg, 'events_count': 'sum'}

    agg_dict = common_cols_map
    agg_dict.update(speed_acc_cols_map)
    agg_dict.update(additional)

    df_quantum[speed_acc_cols] = df_quantum[speed_acc_cols].apply(pd.to_numeric)

    df_sampling = df_quantum.groupby(pd.Grouper(freq=TIME_SAMPLE_FREQ)).agg(agg_dict)

    df_rolling = df_quantum.rolling(TIME_SAMPLE_FREQ, min_periods=1, center=False).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)]) \
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    df_sampling.to_csv(os.path.join(out_path, "bt_sampling_ds_" + str(file[-6:])))
    df_rolling.to_csv(os.path.join(out_path, "bt_rolling_ds_" + str(file[-6:])))


def broadcasts_generate_features(file, out_path):
    print(file)
    df = pd.read_csv(file, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'action', 'data', 'package', 'scheme', 'type', 'user'])

    drop_actions = [
        'android.net.wifi.SCAN_RESULTS',
        'android.bluetooth.device.action.FOUND',
        'android.bluetooth.adapter.action.DISCOVERY_STARTED',
        'android.bluetooth.adapter.action.DISCOVERY_FINISHED'
    ]

    df = df[~df['action'].str.contains('|'.join(drop_actions))]
    df = df.drop(['data', 'package', 'scheme', 'type'], axis=1)

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    df['events_count'] = 1

    VALID_USER = df.iloc[0]['user']
    df['user'] = df['user'].apply(lambda x: 1 if x == VALID_USER else 0)

    df = df.drop(['timestamp'], axis=1)

    df.to_csv(os.path.join(out_path, "broadcasts_ds_" + str(file[-6:])))


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
            wifi_generate_features(os.path.join(u_dir, BASE_WIFI_NAME + '_' + str(i) + ".data"), os.path.join(u_dir, CONN_WIFI_NAME + '_' + str(i) + ".data"), new_path)
            bt_generate_features(os.path.join(u_dir, BASE_BT_NAME + '_' + str(i) + ".data"),
                                   os.path.join(u_dir, LE_BT_NAME + '_' + str(i) + ".data"), new_path)
            broadcasts_generate_features(os.path.join(u_dir, BROADCASTS_NAME + '_' + str(i) + ".data"), new_path)


if __name__ == '__main__':
    main()