import pandas as pd
import random as rnd
import os
from datetime import datetime as dt, timedelta as td
import argparse

BASE_BT_NAME = 'base_bt'
BASE_WIFI_NAME = 'base_wifi'
BROADCASTS_NAME = 'broadcasts'
CONN_WIFI_NAME = 'conn_wifi'
LE_BT_NAME = 'le_bt'
LOCATION_NAME = 'location'

USERS_COUNT = 6
TIME_RANGE = 10
GEN_DF_COUNT = 10

POWER_EVENTS_FILE_NAME = "power.data"

SAMPLES_COUNT = 10
DURATION = 15


GENERATED_FLOW_NAME = "flow"


def convert_timestamp_for_file(file_path):
    print("\tConvert: ", file_path)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    timestamps = [dt.strptime(x.split(';')[0], '%d.%m.%Y_%H:%M:%S.%f') for x in lines]
    users = [int(x.split(';')[-1].replace('\n', '')) for x in lines]

    df = pd.DataFrame({'timestamp' : timestamps, 'user' : users})
    df.index = pd.DatetimeIndex(df.timestamp)

    df['delta'] = (df.timestamp.shift(-1) - df.timestamp).shift(1)

    idx = df.index[df['user'].diff() != 0][-1]
    df.at[idx, 'delta'] = pd.Timedelta(0)

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

    ts = [x.strftime('%d.%m.%Y_%H:%M:%S.%f') for x in df['timestamp']]
    new_lines = []
    for line, timestamp in zip(lines, ts):
        new_line = line.split(';')
        new_line[0] = timestamp
        new_lines.append(';'.join(new_line))

    with open(file_path, 'w') as of:
        of.writelines(new_lines)


def convert_timestamps(path):
    for time_dir in os.listdir(path):
        dir = os.path.join(path, time_dir)
        for user_path in os.listdir(dir):
            flow_path = os.path.join(dir, user_path, GENERATED_FLOW_NAME)
            print("Convert timestamps: ", flow_path)

            for file in os.listdir(flow_path):
                file_path = os.path.join(flow_path, file)
                if os.path.isfile(file_path):
                    convert_timestamp_for_file(file_path)


def merge_different_users_samples(path, samples_count):
    for time_dir in os.listdir(path):
        dir = os.path.join(path, time_dir)
        for user_path in os.listdir(dir):
            print("Merge samples for ", user_path)

            src_user_path = os.path.join(dir, user_path)
            out_user_path = os.path.join(src_user_path, GENERATED_FLOW_NAME)

            if os.path.exists(out_user_path) is False:
                os.makedirs(out_user_path)

            data_files = []
            for file in os.listdir(src_user_path):
                file_path = os.path.join(src_user_path, file)
                if os.path.isfile(file_path):
                    data_files.append(file_path)

            for file in data_files:
                for other_user_path in os.listdir(dir):
                    if other_user_path != user_path:
                        other_path = os.path.join(dir, other_user_path)
                        other_data_files = []
                        for f in os.listdir(other_path):
                            other_file_path = os.path.join(other_path, f)
                            if os.path.isfile(other_file_path):
                                other_data_files.append(other_file_path)

                        for other_file in other_data_files:
                            if os.path.basename(other_file).split('_')[0] == os.path.basename(file).split('_')[0]:
                                print("\tFile: ", file)
                                print("\tOther file: ", other_file)

                                valid_user = os.path.basename(src_user_path).split('_')[1]
                                intruder = os.path.basename(other_user_path).split('_')[1]

                                with open(file, 'r') as ff:
                                    valid_lines = ff.readlines()

                                with open(other_file, 'r') as ff:
                                    intruder_lines = ff.readlines()

                                valid_lines = [x.replace('\n', ';' + valid_user) + '\n' for x in valid_lines]
                                intruder_lines = [x.replace('\n', ';' + intruder) + '\n' for x in intruder_lines]

                                base_name = os.path.basename(file).split('_')[0]
                                index_1 = os.path.basename(file).split('_')[1].replace('.data', '')
                                index_2 = os.path.basename(other_file).split('_')[1].replace('.data', '')
                                file_name = '_'.join([base_name, index_1, index_2, intruder]) + ".data"
                                with open(os.path.join(out_user_path, file_name), 'w') as of:
                                    of.writelines(valid_lines)
                                    of.writelines(intruder_lines)


def generate_samples_for_each_user(src_path, dst_path, samples_count, duration):
    seed = int(rnd.SystemRandom().random() * 10000)
    rnd.seed(a=seed)

    for user_data_dir in os.listdir(src_path):
        print("Generate samples for ", user_data_dir)

        src_user_path = os.path.join(src_path, user_data_dir)
        out_user_path = os.path.join(dst_path, user_data_dir)

        if os.path.exists(out_user_path) is False:
            os.makedirs(out_user_path)

        data_files = []
        for file in os.listdir(src_user_path):
            file_path = os.path.join(src_user_path, file)
            if os.path.isfile(file_path) and file != POWER_EVENTS_FILE_NAME:
                data_files.append(file_path)

        for file in data_files:
            print("\tFile: ", file)
            df = pd.read_csv(file, sep='\n', index_col=False, header=None, low_memory=False)

            df['timestamp'] = df[0].apply(lambda x: x.split(';')[0])
            df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

            df.index = pd.DatetimeIndex(df.timestamp)
            df = df.sort_index()

            df = df.drop(['timestamp'], axis=1)

            for i in range(samples_count):
                print("\t\tSample ", i)
                chosen_file_name = data_files[rnd.randrange(len(data_files))]
                with open(chosen_file_name) as f:
                    lines = f.readlines()

                index = rnd.randrange(len(lines) // 5, 4 * (len(lines) // 5))
                begin_timestamp = dt.strptime(lines[index].split(';')[0], '%d.%m.%Y_%H:%M:%S.%f')
                end_timestamp = begin_timestamp + td(minutes=duration)

                sample = df[begin_timestamp : end_timestamp]

                p = os.path.join(out_user_path, os.path.basename(file).replace(".data", "_" + str(i) + ".data"))
                sample.to_csv(p, sep=';', header=False, index=False)

                with open(p, 'r') as f:
                    lines = f.readlines()

                lines = [x[1:-2] + '\n' for x in lines]
                with open(p, 'w') as f:
                    f.writelines(lines)


def generate_events(src_path, dst_path, samples_count, duration):
    generate_samples_for_each_user(src_path, os.path.join(dst_path, str(duration) + "min"), samples_count, duration)


def main():
    parser = argparse.ArgumentParser(description='Events flow generator')

    parser.add_argument("--src", default=None, type=str, help="Folder with data after filtering")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst

    # generate_events(src_folder, dst_folder, SAMPLES_COUNT, DURATION)
    # merge_different_users_samples(dst_folder, SAMPLES_COUNT)
    convert_timestamps(dst_folder)

if __name__ == '__main__':
    main()
