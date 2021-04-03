import os
import pandas as pd


SAMPLING_FREQs = ['5s', '10s', '30s', '60s', '90s', '120s', '240s', '600s']

FILE_TYPE = "broadcasts"
COMMON_DATA_POSTFIX = ".data"

PIPELINE_PATH = "_gen"
PROCESS_PATH = "_generated"
DATASETS_PATH = "_datasets"


def broadcasts_file_process(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    if os.path.exists(os.path.join(os.getcwd(), PROCESS_PATH)) is False:
        os.mkdir(os.path.join(os.getcwd(), PROCESS_PATH))

    new_file_path = os.path.join(os.getcwd(), PROCESS_PATH, os.path.basename(filepath))

    with open(new_file_path, 'w+', encoding='utf-8') as f:
        for line, i in zip(lines, range(len(lines))):
            f.write(line)

    return {'BASE': new_file_path}


def broadcasts_make_dataframes(file_path, sampling_freq, rolling = False, sampling = True):
    df = pd.read_csv(file_path, sep=';', index_col=False, header=None,
                     low_memory=False, names=['timestamp', 'action', 'data', 'package', 'scheme', 'type'])

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

    df = df.drop(['timestamp'], axis=1)

    index = os.path.basename(file_path).split('_')[-1][0]
    new_file_name = FILE_TYPE + "_" + index + ".csv"
    new_files_dir = os.path.join(os.path.curdir, DATASETS_PATH, sampling_freq)

    if os.path.exists(new_files_dir) is False:
        os.makedirs(new_files_dir)

    df.to_csv(os.path.join(new_files_dir, new_file_name))


def get_broadcasts_files():
    files_list = []
    folder = os.path.join(os.path.curdir, PIPELINE_PATH)
    for file in os.listdir(folder):
        if file.endswith(COMMON_DATA_POSTFIX) and file.find(FILE_TYPE) != -1:
            files_list.append(os.path.join(folder, file))

    return files_list


def broadcasts_pipeline(sampling_freqs):
    for freq in sampling_freqs:
        logs = []
        for file in get_broadcasts_files():
            logs.append(broadcasts_file_process(file))
        for t in logs:
            print(t['BASE'], freq)
            broadcasts_make_dataframes(t['BASE'], freq, True, True)


if __name__ == '__main__':
    broadcasts_pipeline(SAMPLING_FREQs)