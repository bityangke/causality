import xml.etree.ElementTree as ET
import argparse
from os import path
import copy


def read_all_data(xmlfile):
    tree = ET.parse(xmlfile)
    return tree


def write(output_file, tree, start_idx=1, end_idx=1000):
    root = tree.getroot()
    for child in root.findall('item'):
        id = int(child.attrib['id'])
        if id < start_idx or id > end_idx:
            root.remove(child)

    tree.write(output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir", type=str,
        default="/home/shtoshni/Research/causality/tasks/copa/data/orig_split")
    parser.add_argument(
        "-output_dir", type=str,
        default="/home/shtoshni/Research/causality/tasks/copa/data/diff_split")

    hp = parser.parse_args()
    return hp


def main():
    hp = parse_args()
    tree = read_all_data(path.join(hp.data_dir, "copa-all.xml"))
    # Train set
    write(path.join(hp.output_dir, "copa-train.xml"), copy.deepcopy(tree), 1, 500)
    # Dev set
    write(path.join(hp.output_dir, "copa-dev.xml"), copy.deepcopy(tree), 501, 750)
    # # Test set
    write(path.join(hp.output_dir, "copa-test.xml"), copy.deepcopy(tree), 751, 1000)


if __name__ == '__main__':
    main()
