import pandas as pd
import random as rnd
import os
from datetime import datetime as dt

BASE_BT_NAME = 'base_bt'
BASE_WIFI_NAME = 'base_wifi'
BROADCASTS_NAME = 'broadcasts'
CONN_WIFI_NAME = 'conn_wifi'
LE_BT_NAME = 'le_bt'
LOCATION_NAME = 'location'

USERS_COUNT = 6
TIME_RANGE = 10
GEN_DF_COUNT = 10


def make_common_dataframe(df_count, df_type):
    dfs_list = []
    dfs_rows_len_list = []
    for i in range(1, df_count + 1):
        df = pd.read_csv(".\\_events\\" + df_type + "_filtered_" + str(i) + ".data", sep=';', header=None)
        df['user'] = i
        dfs_list.append(df)
        dfs_rows_len_list.append(df.shape[0])

    concat_list = []
    for df in dfs_list:
        concat_list.append(df)

    df = pd.concat(concat_list, ignore_index=True)
    return df


def get_random_series_of_events(df, duration_min):
    while True:
        print(df.columns)
        begin_index = rnd.randrange(len(df))
        begin = df.iloc[begin_index]['timestamp']
        end = begin + pd.Timedelta(duration_min, unit='min')
        events = df[begin:end].copy()
        if len(events) > 20:
            break

    return events


def convert_timestamps(df_):
    df = df_.copy()
    df['delta'] = (df.timestamp.shift(-1) - df.timestamp).shift(1)

    idx = df.index[df['user'].diff() != 0][-1]
    df.at[df.index[idx], 'delta'] = pd.Timedelta(0)

    time_array = []
    current_timestamp = [df.timestamp[0]]

    def add_value_to_array(x, time_array, current_timestamp):
        delta = x
        if pd.isnull(x):
            delta = pd.Timedelta(0)

        current_timestamp[0] += delta
        time_array.append(current_timestamp[0])

    _ = df.delta.apply(lambda x: add_value_to_array(x, time_array, current_timestamp))

    df.timestamp = time_array
    df = df.drop(['delta'], axis=1)

    return df


def events_flow_generator(df_dict, valid_user, duration):
    new_events_dict = {}
    while True:
        intruder = rnd.randrange(1, USERS_COUNT + 1)
        if intruder != valid_user:
            break

    for t, df in df_dict.items():
        print(t)
        valid_user_df = df[df.user == valid_user].copy()
        intruder_df = df[df.user == intruder].copy()
        valid_events = get_random_series_of_events(valid_user_df, duration)
        intruder_events = get_random_series_of_events(intruder_df, duration)
        flow_df = pd.concat([valid_events, intruder_events], ignore_index=True)
        flow_df = convert_timestamps(flow_df)
        new_d = {t: flow_df}
        new_events_dict.update(new_d)

    return new_events_dict


def main():
    df_base_bt = make_common_dataframe(USERS_COUNT, BASE_BT_NAME)
    df_base_wifi = make_common_dataframe(USERS_COUNT, BASE_WIFI_NAME)
    df_broadcasts = make_common_dataframe(USERS_COUNT, BROADCASTS_NAME)
    df_conn_wifi = make_common_dataframe(USERS_COUNT, CONN_WIFI_NAME)
    df_le_bt = make_common_dataframe(USERS_COUNT, LE_BT_NAME)
    df_location = make_common_dataframe(USERS_COUNT, LOCATION_NAME)

    seed = int(rnd.SystemRandom().random() * 10000)
    rnd.seed(a=seed)

    df_dict = {
        BASE_BT_NAME: df_base_bt,
        BASE_WIFI_NAME: df_base_wifi,
        BROADCASTS_NAME: df_broadcasts,
        CONN_WIFI_NAME: df_conn_wifi,
        LE_BT_NAME: df_le_bt,
        LOCATION_NAME: df_location
    }

    tmp = {}
    for key, df in df_dict.items():
        print(key)
        new_df = df.rename(columns={0: "timestamp"})
        new_df.timestamp = new_df.timestamp.apply(lambda x: dt.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        new_df.index = pd.DatetimeIndex(new_df.timestamp)
        new_df = new_df.sort_index()
        tmp.update({key: new_df})

    df_dict.update(tmp)

    for i in range(1, USERS_COUNT + 1):
        for j in range(GEN_DF_COUNT):
            events_dict = events_flow_generator(df_dict, i, TIME_RANGE)
            path = ".\\_events\\_generated\\valid_user_" + str(i)
            os.makedirs(path, exist_ok=True)
            for key, value in events_dict.items():
                value.to_csv(os.path.join(path, key + "_" + str(j) + ".data"), sep=';', header=False, index=False)


if __name__ == '__main__':
    main()
