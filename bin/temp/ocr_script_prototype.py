import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

from src.lib.ocr import print_pages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--pdf_file', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    args = parser.parse_args()
    print_pages(args.pdf_file, args.output_file)
