import os
import zipfile
import argparse
import pandas as pd
from datetime import datetime as dt
import random as rnd

BROADCASTS_FILE_NAME = "broadcasts.data"
POWER_EVENTS_FILE_NAME = "power.data"
TMP_DIR_NAME = "tmp"


def create_periods_file(data_folder):
    df = pd.read_csv(os.path.join(os.path.abspath(data_folder), BROADCASTS_FILE_NAME), sep=';', index_col=False,
                     header=None, low_memory=False, names=['timestamp', 'action', 'data', 'package', 'scheme', 'type'])

    SCREEN_ON_EVENT = 'android.intent.action.SCREEN_ON'
    SCREEN_OFF_EVENT = 'android.intent.action.SCREEN_OFF'

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    power_events_df = df[df['action'].str.contains('|'.join(
        [SCREEN_ON_EVENT,
         SCREEN_OFF_EVENT]))]

    power_events_df = power_events_df.append(
        {
            'timestamp': df.iloc[0][0],
            'action': SCREEN_ON_EVENT,
            'data': df.iloc[0][2]
        }, ignore_index=True)

    power_events_df.index = pd.DatetimeIndex(power_events_df.timestamp)
    power_events_df = power_events_df.sort_index()

    with open(os.path.join(os.path.abspath(data_folder), POWER_EVENTS_FILE_NAME), 'a') as f:
        time_array = []
        for i in range(len(power_events_df)):
            action = power_events_df['action'].iloc[i]
            timestamp = power_events_df['timestamp'].iloc[i]

            if action == SCREEN_ON_EVENT:
                on_time = timestamp
            else:
                time_array.append([on_time, timestamp])

        for on, off in time_array:
            if (off - on).total_seconds() <= 60 * 60:
                f.write(str(on) + ';' + str(off) + '\n')





def filter_logs_by_time(dst_folder):
    for user_data_dir in os.listdir(os.path.abspath(dst_folder)):
        create_periods_file(os.path.join(os.path.abspath(dst_folder), user_data_dir))
        for file in os.listdir(os.path.join(os.path.abspath(dst_folder), user_data_dir)):
            if os.path.isfile(os.path.join(os.path.abspath(dst_folder), user_data_dir, file)) and file != POWER_EVENTS_FILE_NAME:
                filter_logs(os.path.join(os.path.abspath(dst_folder), user_data_dir, file), user_data_dir[-1:], dst_folder)


def bt_file_split(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    dir = os.path.dirname(os.path.abspath(filepath))
    new_file_path = os.path.join(dir, "_".join(["base", os.path.basename(filepath)]))
    new_le_file_path = os.path.join(dir, "_".join(["le", os.path.basename(filepath)]))

    with open(new_file_path, 'w+', encoding='utf-8') as f:
        for line, i in zip(lines, range(len(lines))):
            if line.find(';LE;') == -1:
                f.write(line)

    with open(new_le_file_path, 'w+', encoding='utf-8') as f:
        for line, i in zip(lines, range(len(lines))):
            if line.find(';LE;') != -1:
                f.write(line)

    os.remove(filepath)

    return {'BASE': new_file_path, 'LE': new_le_file_path}


def wifi_file_split(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    dir = os.path.dirname(os.path.abspath(filepath))
    new_file_path = os.path.join(dir, "_".join(["base", os.path.basename(filepath)]))
    new_conn_file_path = os.path.join(dir, "_".join(["conn", os.path.basename(filepath)]))

    with open(new_file_path, 'w+', encoding='utf-8') as f:
        for line, i in zip(lines, range(len(lines))):
            if line.find(';CONN;') == -1:
                f.write(line)

    with open(new_conn_file_path, 'w+', encoding='utf-8') as f:
        for line, i in zip(lines, range(len(lines))):
            if line.find(';CONN;') != -1:
                f.write(line)

    os.remove(filepath)

    return {'BASE': new_file_path, 'CONN': new_conn_file_path}


def split_files(dst_folder):
    for user_data_dir in os.listdir(os.path.abspath(dst_folder)):
        for file in os.listdir(os.path.join(os.path.abspath(dst_folder), user_data_dir)):
            if os.path.isfile(os.path.join(os.path.abspath(dst_folder), user_data_dir, file)) and file.find('bt') != -1:
                bt_file_split(os.path.join(os.path.abspath(dst_folder), user_data_dir, file))
            elif os.path.isfile(os.path.join(os.path.abspath(dst_folder), user_data_dir, file)) and file.find('wifi') != -1:
                wifi_file_split(os.path.join(os.path.abspath(dst_folder), user_data_dir, file))


BASE_BT_NAME = 'base_bt'
BASE_WIFI_NAME = 'base_wifi'
BROADCASTS_NAME = 'broadcasts'
CONN_WIFI_NAME = 'conn_wifi'
LE_BT_NAME = 'le_bt'
LOCATION_NAME = 'location'


USERS_COUNT = 8
TIME_RANGE = 10
GEN_DF_COUNT = 10


def make_common_dataframe(dst_folder, df_count, df_type):
    dfs_list = []
    dfs_rows_len_list = []
    for i in range(1, df_count + 1):
        path = os.path.join(dst_folder, "user_" + str(i))
        df = pd.read_csv(os.path.join(path, df_type + ".data"), sep=';', header=None)
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


def events_generator(dst_folder):
    df_base_bt = make_common_dataframe(dst_folder, USERS_COUNT, BASE_BT_NAME)
    df_base_wifi = make_common_dataframe(dst_folder, USERS_COUNT, BASE_WIFI_NAME)
    df_broadcasts = make_common_dataframe(dst_folder, USERS_COUNT, BROADCASTS_NAME)
    df_conn_wifi = make_common_dataframe(dst_folder, USERS_COUNT, CONN_WIFI_NAME)
    df_le_bt = make_common_dataframe(dst_folder, USERS_COUNT, LE_BT_NAME)
    df_location = make_common_dataframe(dst_folder, USERS_COUNT, LOCATION_NAME)

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


def main():
    global TMP_DIR_NAME
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument("--data", default=None, type=str, help="Raw zip data folder")
    parser.add_argument("--folder", default=os.path.curdir, type=str, help="Destination folder")
    parser.add_argument("--gen", action="store_true", help="Generate flow")

    args = parser.parse_args()

    src_folder = args.data
    dst_folder = args.folder

    should_generate = args.gen

    if os.path.exists(os.path.join(os.getcwd(), TMP_DIR_NAME)) is False:
        os.mkdir(os.path.join(os.getcwd(), TMP_DIR_NAME))

    TMP_DIR_NAME = os.path.join(os.getcwd(), TMP_DIR_NAME)

    user_data_folders = [os.path.join(src_folder, x) for x in os.listdir(src_folder)]



    split_files(dst_folder)
    if should_generate:
        events_generator(dst_folder)


    os.rmdir(TMP_DIR_NAME)


if __name__ == '__main__':
    main()
