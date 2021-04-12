import pandas as pd
import numpy as np
from datetime import datetime as dt
import scipy.stats as stats
from scipy.spatial import distance
from geopy.distance import distance as geodist
import argparse
import os

POWER_EVENTS_FILE_NAME = "power.data"


def generate_location_features(src_path, dst_path_rolling, dst_path_sampling, freq):
    df = pd.read_csv(src_path, sep=';', index_col=False, header=None, low_memory=False,
                     names=['timestamp', 'accuracy', 'altitude', 'latitude', 'longitude', 'user'])

    if len(df) == 0:
        return

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

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

    df = df.drop(['prev_latitude', 'prev_longitude', 'prev_altitude', 'timestamp', 'prev_timestamp'], axis=1)

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

    df_sampling = df.groupby(pd.Grouper(freq=freq)).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling = df.rolling(freq, min_periods=1, center=False).agg(agg_dict)

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    df_sampling.to_csv(dst_path_sampling)
    df_rolling.to_csv(dst_path_rolling)


def generate_wifi_features(src_path, src_path_conn, dst_path_rolling, dst_path_sampling, freq, window):
    df = pd.read_csv(src_path, sep=';', index_col=False, header=None, low_memory=False,
                     names=['timestamp', 'uuid', 'bssid', 'chwidth', 'freq', 'level', 'user'])

    if len(df) == 0:
        return

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))
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
                              'count': 'sum', 'events_count': 'sum', 'user': user_agg}

    df_quantum = df.groupby(['timestamp', 'uuid'], as_index=True).agg(all_func_dicts_quantum)

    df_quantum = df_quantum.reset_index()
    df_quantum.index = pd.DatetimeIndex(df_quantum.timestamp)

    df_quantum = df_quantum[df_quantum['count'] != 0]

    df_conn = pd.read_csv(src_path_conn, sep=';', index_col=False, header=None, low_memory=False,
                          names=['timestamp', '1', 'bssid', '2', '3', '4', '5', 'level', '6'])

    if len(df_conn) != 0:
        df_conn['timestamp'] = df_conn['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))
        df_conn.index = pd.DatetimeIndex(df_conn.timestamp)
        df_conn = df_conn.sort_index()

        def get_level_from_row(row):
            bssid = df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['bssid']
            if str(bssid) == 'nan' or str(bssid) == 'null' or str(bssid) == '':
                return 0

            level = df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['level']
            time = df_conn.iloc[df_conn.index.get_loc(row.name, method='nearest')]['timestamp']
            return level if abs((time - row.name).total_seconds()) <= 10 else 0

        df_conn = df_conn.loc[~df_conn.index.duplicated(keep='first')]
        df_quantum['conn_level'] = df_quantum.apply(lambda row: get_level_from_row(row), axis=1)
    else:
        df_quantum['conn_level'] = 0

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

    occur_and_level_columns_map = [
        ("bssid_level", "occured_nets_count", window, get_occured_nets_count),
        ("bssid_level", "disappeared_nets_count", window, get_disappeared_nets_count),
        ("bssid_level", "jaccard_index", window, get_jaccard_index),
        ("bssid_level", "occur_speed", window, get_occur_speed),
        ("bssid_level", "level_speed", window, get_level_speed)
    ]

    for (col, new_col, wnd, func) in occur_and_level_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, wnd, func)

    def get_conn_level_speed(row, prev_col, curr_col):
        return row[curr_col] - row[prev_col]

    single_columns_map = [
        ("conn_level", "conn_level_speed", window, get_conn_level_speed),
        ("count", "count_speed", window, get_conn_level_speed)
    ]

    for (col, new_col, wnd, func) in single_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, wnd, func)

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

    df_quantum['freq'] = df_quantum.apply(lambda row: str_mean(row['freq']), axis=1)
    df_quantum['level'] = df_quantum.apply(lambda row: str_mean(row['level']), axis=1)

    cols_for_drop = []
    names = [
        "occured_nets_count",
        "disappeared_nets_count",
        "jaccard_index",
        "occur_speed",
        "count_speed",
        "conn_level_speed",
        "level_speed",
        "count_speed"
    ]

    for i in range(1, window + 1):
        for name in names:
            cols_for_drop.append('_'.join([name, str(i)]))

    df_quantum = df_quantum.drop(['bssid_level', 'timestamp', 'uuid'], axis=1)
    df_quantum = df_quantum.drop(cols_for_drop, axis=1)

    def user_agg(col):
        if (col == 1).all():
            return 1
        else:
            return 0

    common_cols = [x for x in df_quantum.columns[0:6] if x != 'user' and x != 'events_count']
    speed_acc_cols = df_quantum.columns[6:]

    common_funcs_list = [mean, var, median, skew, kurt]
    special_funcs_list = [mean, pd.DataFrame.mad, skew]

    common_cols_map = {col: common_funcs_list for col in common_cols}
    speed_acc_cols_map = {col: special_funcs_list for col in speed_acc_cols}

    additional = {'user': user_agg, 'events_count': 'sum'}

    agg_dict = common_cols_map
    agg_dict.update(speed_acc_cols_map)
    agg_dict.update(additional)

    df_quantum[speed_acc_cols] = df_quantum[speed_acc_cols].apply(pd.to_numeric)

    df_sampling = df_quantum.groupby(pd.Grouper(freq=freq)).agg(agg_dict)

    df_rolling = df_quantum.rolling(freq, min_periods=1, center=False).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    df_sampling.to_csv(dst_path_sampling)
    df_rolling.to_csv(dst_path_rolling)


def generate_bt_features(src_path, src_path_le, dst_path_rolling, dst_path_sampling, freq, window):
    df = pd.read_csv(src_path, sep=';', index_col=False, header=None, low_memory=False,
                     names=['timestamp', 'action', 'bssid', 'major_class', 'class', 'bond_state', 'type', 'user'])

    if len(df) == 0:
        return

    df = df[df['action'] == 'android.bluetooth.device.action.FOUND']

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))
    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    VALID_USER = df.iloc[0]['user']
    df['events_count'] = 1

    df = df.drop(['timestamp', 'action', 'class', 'major_class', 'bond_state', 'type'], axis=1)

    bssid_map = {bssid.replace(' ', ''): idx for bssid, idx in zip(df.bssid.unique(), range(len(df.bssid.unique())))}

    df.bssid = df.bssid.apply(lambda x: str(x).replace(' ', ''))
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

    cat_columns = df.columns[1:1 + one_hot_columns_count]
    cat_columns_map = {col: 'mean' for col in cat_columns}

    all_func_dicts_quantum = {'bssid': agg_bssid_col, 'count': 'sum', 'user': user_agg, 'events_count': 'sum'}
    all_func_dicts_quantum.update(cat_columns_map)

    df_quantum = df.groupby(pd.Grouper(freq='5s'), as_index=True).agg(all_func_dicts_quantum)

    df_quantum = df_quantum.reset_index()
    df_quantum.index = pd.DatetimeIndex(df_quantum.timestamp)

    df_quantum = df_quantum.dropna()

    df_le = pd.read_csv(src_path_le, sep=';', index_col=False, header=None, low_memory=False,
                        names=['timestamp', '1', '2', '3', 'level', '3', 'connectable', '4'])

    if len(df_le) != 0:
        df_le['timestamp'] = df_le['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))
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

        df_le = df_le.loc[~df_le.index.duplicated(keep='first')]

        df_quantum['le_connectable'] = df_quantum.apply(lambda row: get_le_conn_status_from_row(row), axis=1)
        df_quantum['le_level'] = df_quantum.apply(lambda row: get_le_level_from_row(row), axis=1)

    else:
        df_quantum['le_connectable'] = 0
        df_quantum['le_level'] = 0

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

    occur_and_level_columns_map = [
        ("bssid", "occured_devices_count", window, get_occured_nets_count),
        ("bssid", "disappeared_devices_count", window, get_disappeared_nets_count),
        ("bssid", "jaccard_index", window, get_jaccard_index),
        ("bssid", "occur_speed", window, get_occur_speed)
    ]

    for (col, new_col, wnd, func) in occur_and_level_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, wnd, func)

    def get_conn_level_speed(row, prev_col, curr_col):
        return row[curr_col] - row[prev_col]

    single_columns_map = [
        ("count", "count_speed", window, get_conn_level_speed)
    ]

    for (col, new_col, wnd, func) in single_columns_map:
        df_quantum = calc_single_cols_in_window(df_quantum, col, new_col, wnd, func)

    def agg_str(col):
        all_freq = col.str.cat(sep=',')
        return string2array(all_freq)

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

    for i in range(1, window + 1):
        for name in names:
            cols_for_drop.append('_'.join([name, str(i)]))

    df_quantum = df_quantum.drop(['bssid', 'timestamp'], axis=1)
    df_quantum = df_quantum.drop(cols_for_drop, axis=1)

    def user_agg(col):
        if (col == 1).all():
            return 1
        else:
            return 0

    common_cols = [x for x in df_quantum.columns[:one_hot_columns_count + 5] if x != 'user' and x != 'events_count']
    speed_acc_cols = df_quantum.columns[one_hot_columns_count + 5:]

    common_funcs_list = [mean, var, median, skew, kurt]
    special_funcs_list = [mean, pd.DataFrame.mad, skew]

    common_cols_map = {col: common_funcs_list for col in common_cols}
    speed_acc_cols_map = {col: special_funcs_list for col in speed_acc_cols}

    additional = {'user': user_agg, 'events_count': 'sum'}

    agg_dict = common_cols_map
    agg_dict.update(speed_acc_cols_map)
    agg_dict.update(additional)

    df_quantum[speed_acc_cols] = df_quantum[speed_acc_cols].apply(pd.to_numeric)

    df_sampling = df_quantum.groupby(pd.Grouper(freq=freq)).agg(agg_dict)

    df_rolling = df_quantum.rolling(freq, min_periods=1, center=False).agg(agg_dict)

    df_sampling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                           for (high_level_name, low_level_name) in df_sampling.columns.values]

    df_rolling.columns = ["_".join([str(high_level_name), str(low_level_name)])
                          for (high_level_name, low_level_name) in df_rolling.columns.values]

    df_sampling = df_sampling.dropna()
    df_sampling = df_sampling.fillna(0)

    df_rolling = df_rolling.dropna()
    df_rolling = df_rolling.fillna(0)

    df_sampling.to_csv(dst_path_sampling)
    df_rolling.to_csv(dst_path_rolling)


def generate_broadcast_features(src_path, dst_path_rolling, dst_path_sampling):
    df = pd.read_csv(src_path, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'action', 'data', 'package', 'scheme', 'type', 'user'])

    if len(df) == 0:
        return

    drop_actions = [
        'android.net.wifi.SCAN_RESULTS',
        'android.bluetooth.device.action.FOUND',
        'android.bluetooth.adapter.action.DISCOVERY_STARTED',
        'android.bluetooth.adapter.action.DISCOVERY_FINISHED'
    ]

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

    df = df[~df['action'].str.contains('|'.join(drop_actions))]
    df = df.drop(['data', 'package', 'scheme', 'type'], axis=1)

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    df['events_count'] = 1
    VALID_USER = VALID_USER = df.iloc[0]['user']
    df['user'] = df['user'].apply(lambda x: 1 if x == VALID_USER else 0)

    df = df.drop(['timestamp'], axis=1)

    df.to_csv(dst_path_rolling)
    df.to_csv(dst_path_sampling)


def generate_features(src, dst, freq, window):
    for time_dir in os.listdir(src):
        dir = os.path.join(src, time_dir)
        for user_path in os.listdir(dir):
            src_user_path = os.path.join(dir, user_path, "flow")
            out_user_sampling_path = os.path.join(dst, time_dir, "sampling", freq, user_path)
            out_user_rolling_path = os.path.join(dst, time_dir, "rolling", freq, user_path)

            if os.path.exists(out_user_sampling_path) is False:
                os.makedirs(out_user_sampling_path)

            if os.path.exists(out_user_rolling_path) is False:
                os.makedirs(out_user_rolling_path)

            print("Generate features for ", user_path)

            for file in os.listdir(src_user_path):
                if os.path.isfile(os.path.join(src_user_path, file)) and file.find(POWER_EVENTS_FILE_NAME) == -1:
                    prefix = file[:-11]
                    postfix = file[-11:]

                    # # # # # # # # # # # #
                    # if postfix[1] == '0' and postfix[3] == '0':
                    # # # # # # # # # # # #

                    if prefix == "base_wifi":
                        wifi_path = os.path.join(src_user_path, prefix + postfix)
                        wifi_conn = os.path.join(src_user_path, "conn_wifi" + postfix)
                        wifi_sampling_out = os.path.join(out_user_sampling_path, "wifi" + postfix + ".csv")
                        wifi_rolling_out = os.path.join(out_user_rolling_path, "wifi" + postfix + ".csv")

                        print("\tGenerate WIFI: ", wifi_path)
                        generate_wifi_features(wifi_path, wifi_conn, wifi_rolling_out, wifi_sampling_out, freq, window)

                    if prefix == "base_bt":
                        bt_path = os.path.join(src_user_path, prefix + postfix)
                        bt_le = os.path.join(src_user_path, "le_bt" + postfix)
                        bt_sampling_out = os.path.join(out_user_sampling_path, "bt" + postfix + ".csv")
                        bt_rolling_out = os.path.join(out_user_rolling_path, "bt" + postfix + ".csv")

                        print("\tGenerate BT: ", bt_path)
                        generate_bt_features(bt_path, bt_le, bt_rolling_out, bt_sampling_out, freq, window)

                    if prefix == "broadcasts":
                        broadcasts_path = os.path.join(src_user_path, prefix + postfix)
                        broadcasts_sampling_out = os.path.join(out_user_sampling_path, "broadcasts" + postfix + ".csv")
                        broadcasts_rolling_out = os.path.join(out_user_rolling_path, "broadcasts" + postfix + ".csv")

                        print("\tGenerate BROADCASTS: ", broadcasts_path)
                        generate_broadcast_features(broadcasts_path, broadcasts_rolling_out, broadcasts_sampling_out)

                    if prefix == "location":
                        location_path = os.path.join(src_user_path, prefix + postfix)
                        location_sampling_out = os.path.join(out_user_sampling_path, "location" + postfix + ".csv")
                        location_rolling_out = os.path.join(out_user_rolling_path, "location" + postfix + ".csv")

                        print("\tGenerate LOCATION: ", location_path)
                        generate_location_features(location_path, location_rolling_out, location_sampling_out, freq)


def main():
    parser = argparse.ArgumentParser(description='Features generator for events')

    parser.add_argument("--src", default=None, type=str, help="Folder with data")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")
    parser.add_argument("--wnd", default=None, type=str, help="Window size")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst
    freq = args.wnd

    OTHER_WINDOW_SIZE = 3

    generate_features(src_folder, dst_folder, freq, OTHER_WINDOW_SIZE)


if __name__ == '__main__':
    main()
