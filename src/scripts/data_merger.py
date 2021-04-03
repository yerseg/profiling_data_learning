import argparse
import os


def merge_files_in_folder(src_path, dst_path):
    try:
        log = ''
        out_file_name = ".".join([os.path.basename(src_path), 'data'])

        if os.path.exists(dst_path) is False:
            os.makedirs(dst_path)

        with open(os.path.join(dst_path, out_file_name), 'w+', encoding='utf-8') as out_file:
            for file, i in zip(os.listdir(src_path), range(len(os.listdir(src_path)))):
                if file != out_file_name:
                    with open(os.path.join(src_path, file), 'r', encoding='utf-8') as f:
                        out_file.writelines(f.readlines())
                    log += '\n' + file
    except Exception as ex:
        print(ex)
    finally:
        print(log)


def merge_files(src_folder, dst_folder):
    for user_data_dir in os.listdir(os.path.abspath(src_folder)):
        for category_dir in os.listdir(os.path.join(os.path.abspath(src_folder), user_data_dir)):
            merge_files_in_folder(os.path.join(os.path.abspath(src_folder), user_data_dir, category_dir),
                                  os.path.join(os.path.abspath(dst_folder), user_data_dir))


def main():
    parser = argparse.ArgumentParser(description='Data merging tool')

    parser.add_argument("--src", default=None, type=str, help="Folder with data after zip unpacking")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst

    merge_files(src_folder, dst_folder)


if __name__ == '__main__':
    main()
