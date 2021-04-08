import argparse
import os
import pandas as pd
from datetime import datetime as dt

BROADCASTS_NAME = "broadcasts"
BROADCASTS_FILE_NAME = "".join([BROADCASTS_NAME, ".data"])
POWER_EVENTS_FILE_NAME = "power.data"


def create_periods_file(src_folder, data_folder):
    df = pd.read_csv(os.path.join(os.path.abspath(src_folder), BROADCASTS_FILE_NAME), sep=';', index_col=False,
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


def filter_logs(src_path, dst_path, file_name):
    df = pd.read_csv(os.path.join(src_path, file_name), sep='\n', index_col=False,
                     header=None, low_memory=False)

    print("Filter logs: ", src_path, ' ', file_name)

    df['timestamp'] = df[0].apply(lambda x: x.split(';')[0])
    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    df = df.drop(['timestamp'], axis=1)

    time_array = []
    with open(os.path.join(dst_path, POWER_EVENTS_FILE_NAME), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(';')
            time_array.append([tmp[0], tmp[1].replace('\n', '')])

    df_parts = []
    for on, off in time_array:
        df_parts.append(df.loc[pd.Timestamp(on): pd.Timestamp(off)])

    new_df = pd.concat(df_parts)
    p = os.path.join(dst_path, file_name)
    new_df.to_csv(p, sep=';', header=False, index=False)

    with open(p, 'r') as f:
        lines = f.readlines()

    lines = [x[1:-2] + '\n' for x in lines]
    with open(p, 'w') as f:
        f.writelines(lines)


def filter_logs_by_time(src_folder, dst_folder):
    for user_data_dir in os.listdir(os.path.abspath(src_folder)):
        src_path = os.path.join(os.path.abspath(src_folder), user_data_dir)
        out_path = os.path.join(os.path.abspath(dst_folder), user_data_dir)

        if os.path.exists(out_path) is False:
            os.makedirs(out_path)

        create_periods_file(src_path, out_path)
        for file in os.listdir(src_path):
            if os.path.isfile(os.path.join(src_path, file)) and file != POWER_EVENTS_FILE_NAME:
                filter_logs(src_path, out_path, file)


def main():
    parser = argparse.ArgumentParser(description='Data filtering by time')

    parser.add_argument("--src", default=None, type=str, help="Folder with data after zip unpacking")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst

    filter_logs_by_time(src_folder, dst_folder)


if __name__ == '__main__':
    main()
