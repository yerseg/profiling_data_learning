import os
import argparse


def bt_file_split(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    le_lines = [x for x in lines if x.find(';LE;') != -1]
    base_lines = [x for x in lines if x.find(';LE;') == -1]

    if len(os.path.basename(filepath).split('_')) == 1:
        return "base_bt.data", "le_bt.data", base_lines, le_lines

    name_postfix = '_'.join(os.path.basename(filepath).split('_')[1:])

    return "base_bt_" + name_postfix, "le_bt_" + name_postfix, base_lines, le_lines


def wifi_file_split(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    conn_lines = [x for x in lines if x.find(';CONN;') != -1]
    base_lines = [x for x in lines if x.find(';CONN;') == -1]

    if len(os.path.basename(filepath).split('_')) == 1:
        return "base_wifi.data", "conn_wifi.data", base_lines, conn_lines

    name_postfix = '_'.join(os.path.basename(filepath).split('_')[1:])

    return "base_wifi_" + name_postfix, "conn_wifi_" + name_postfix, base_lines, conn_lines


def split_files(src, dst):
    for subdir, dirs, files in os.walk(src):
        for file in files:
            file_path = os.path.join(subdir, file)
            print("Split file: ", file_path)
            if os.path.isfile(file_path) and file.find('bt') != -1:
                name1, name2, lines1, lines2 = bt_file_split(file_path)
            elif os.path.isfile(file_path) and file.find('wifi') != -1:
                name1, name2, lines1, lines2 = wifi_file_split(file_path)
            elif os.path.isfile(file_path):
                name1 = os.path.basename(file_path)
                name2 = ''
                with open(file_path, 'r') as f:
                    lines1 = f.readlines()
                lines2 = ''

            new_subdir = subdir.replace(src, dst)
            if os.path.exists(new_subdir) is False:
                os.makedirs(new_subdir)

            with open(os.path.join(new_subdir, name1), 'w') as f:
                f.writelines(lines1)

            if name2 != '':
                with open(os.path.join(new_subdir, name2), 'w') as f:
                    f.writelines(lines2)


def main():
    parser = argparse.ArgumentParser(description='File splitter')

    parser.add_argument("--src", default=None, type=str, help="Folder with data")
    parser.add_argument("--dst", default=None, type=str, help="Destination folder")

    args = parser.parse_args()
    src_folder = args.src
    dst_folder = args.dst

    split_files(src_folder, dst_folder)


if __name__ == '__main__':
    main()
