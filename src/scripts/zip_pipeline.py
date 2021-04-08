import os
import zipfile
import argparse
import pandas as pd
from datetime import datetime as dt
import random as rnd

BROADCASTS_FILE_NAME = "broadcasts.data"
POWER_EVENTS_FILE_NAME = "power.data"
TMP_DIR_NAME = "tmp"





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



if __name__ == '__main__':
    main()
