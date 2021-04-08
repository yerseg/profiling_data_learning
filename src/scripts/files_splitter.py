import os
import argparse


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


def split_files(src, dst):
    for subdir, dirs, files in os.walk(src):
        for file in files:
            file_path = os.path.join(subdir, file)
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