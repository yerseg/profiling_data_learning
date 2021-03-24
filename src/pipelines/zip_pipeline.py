import os
import zipfile
import argparse
import pandas as pd
from datetime import datetime as dt

BROADCASTS_FILE_NAME = "broadcasts.data"
POWER_EVENTS_FILE_NAME = "power.data"
TMP_DIR_NAME = "tmp"


def unpack_zip_archives(path):
    with zipfile.ZipFile(path, 'r') as zip_:
        zip_.extractall(TMP_DIR_NAME)


def generate_name(filename, i):
    name_parts = filename.split('.')
    category = name_parts[0]
    name_parts[0] += '_' + str(i)
    return category, ".".join(name_parts)


def process_zips(path, dst_folder):
    path = os.path.join(os.getcwd(), path)
    for zip_file, i in zip(os.listdir(path), range(len(os.listdir(path)))):
        unpack_zip_archives(os.path.join(path, zip_file))
        for filename in os.listdir(TMP_DIR_NAME):
            (category, file) = generate_name(filename, i)
            dst_path = os.path.abspath(dst_folder)
            if os.path.exists(os.path.join(dst_path, os.path.basename(path), category)) is False:
                os.makedirs(os.path.join(dst_path, os.path.basename(path), category))
            os.rename(os.path.join(TMP_DIR_NAME, filename),
                      os.path.join(dst_path, os.path.basename(path), category, file))


def process_user_data(user_data_path_list, dst_folder):
    for path in user_data_path_list:
        process_zips(path, dst_folder)


def merge_files_in_folder(folder_path, bins=1):
    try:
        log = ''
        out_file_name = ".".join([os.path.basename(folder_path), 'data'])
        with open(os.path.join(folder_path, os.path.pardir, out_file_name), 'w+', encoding='utf-8') as out_file:
            for file, i in zip(os.listdir(folder_path), range(len(os.listdir(folder_path)))):
                if file != out_file_name:
                    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                        out_file.writelines(f.readlines())
                    log += '...' + file
    except Exception as ex:
        print(ex)
    finally:
        print(log)


def merge_files(dst_folder, bins=1):
    for user_data_dir in os.listdir(os.path.abspath(dst_folder)):
        for category_dir in os.listdir(os.path.join(os.path.abspath(dst_folder), user_data_dir)):
            merge_files_in_folder(os.path.join(os.path.abspath(dst_folder), user_data_dir, category_dir))


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


def filter_logs(file_name, num, dst_folder):
    df = pd.read_csv(file_name, sep=';', index_col=False,
                     header=None, low_memory=False)

    df = df.rename(columns={0: "timestamp"})

    df['timestamp'] = df['timestamp'].apply(lambda x: dt.strptime(x, '%d.%m.%Y_%H:%M:%S.%f'))

    df.index = pd.DatetimeIndex(df.timestamp)
    df = df.sort_index()

    df = df.drop(['timestamp'], axis=1)

    time_array = []
    with open(os.path.join(os.path.abspath(os.path.dirname(file_name)), POWER_EVENTS_FILE_NAME), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(';')
            time_array.append([tmp[0], tmp[1].replace('\n', '')])

    df_parts = []
    for on, off in time_array:
        df_parts.append(df.loc[pd.Timestamp(on): pd.Timestamp(off)])

    new_df = pd.concat(df_parts)
    p = os.path.join(dst_folder, os.path.basename(file_name.replace(".data", "_filtered_" + num + ".data")))
    new_df.to_csv(p, sep=';', header=False)


def filter_logs_by_time(dst_folder):
    for user_data_dir in os.listdir(os.path.abspath(dst_folder)):
        create_periods_file(os.path.join(os.path.abspath(dst_folder), user_data_dir))
        for category_dir in os.listdir(os.path.join(os.path.abspath(dst_folder), user_data_dir)):
            if os.path.isfile(os.path.join(os.path.abspath(dst_folder), user_data_dir, category_dir)):
                continue

            filter_logs(os.path.join(os.path.abspath(dst_folder), user_data_dir, category_dir + ".data"),
                        user_data_dir[-1:], dst_folder)


def main():
    global TMP_DIR_NAME
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument("--data", default=None, type=str, help="Raw zip data folder")
    parser.add_argument("--folder", default=os.path.curdir, type=str, help="Destination folder")

    args = parser.parse_args()

    src_folder = args.data
    dst_folder = args.folder

    if os.path.exists(os.path.join(os.getcwd(), TMP_DIR_NAME)) is False:
        os.mkdir(os.path.join(os.getcwd(), TMP_DIR_NAME))

    TMP_DIR_NAME = os.path.join(os.getcwd(), TMP_DIR_NAME)

    user_data_folders = [os.path.join(src_folder, x) for x in os.listdir(src_folder)]
    process_user_data(user_data_folders, dst_folder)
    merge_files(dst_folder)
    filter_logs_by_time(dst_folder)

    os.rmdir(TMP_DIR_NAME)


if __name__ == '__main__':
    main()
