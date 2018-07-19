import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from wsi.data.annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default='../../jsons/test/', metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')
parser.add_argument('json_path', default='../../jsons/test/json/', metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')


def run(args):

    filenames = os.listdir(args.xml_path)
    for filename in filenames:
        if '.xml' in filename:
            print(filename)
            print(filename[:-4])
            Formatter.camelyon16xml2json(args.xml_path+filename, args.json_path+filename[:-4]+'.json')


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
