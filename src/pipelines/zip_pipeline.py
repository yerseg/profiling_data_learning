import os
import zipfile
import argparse

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

    os.rmdir(TMP_DIR_NAME)


if __name__ == '__main__':
    main()
