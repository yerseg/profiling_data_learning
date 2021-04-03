import argparse
import os
import zipfile

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


def main():
    global TMP_DIR_NAME
    parser = argparse.ArgumentParser(description='Zip data unpacker')

    parser.add_argument("--src", default=None, type=str, help="Raw zip data source folder")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst

    if os.path.exists(os.path.join(os.getcwd(), TMP_DIR_NAME)) is False:
        os.makedirs(os.path.join(os.getcwd(), TMP_DIR_NAME))

    TMP_DIR_NAME = os.path.join(os.getcwd(), TMP_DIR_NAME)

    user_data_folders = [os.path.join(src_folder, x) for x in os.listdir(src_folder)]
    process_user_data(user_data_folders, dst_folder)

    os.rmdir(TMP_DIR_NAME)


if __name__ == '__main__':
    main()
