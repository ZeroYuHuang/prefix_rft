import sys
import json
import argparse
import math
from pathlib import Path

def split_jsonl_file(input_file, n_splits, output_prefix):
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    lines_per_file = math.ceil(total_lines / n_splits)

    print(f"Total lines in input file: {total_lines}")
    print(f"Lines per split file: {lines_per_file}")

    # Split lines and write to output files
    for i in range(n_splits):
        start_index = i * lines_per_file
        end_index = min(start_index + lines_per_file, total_lines)
        split_lines = lines[start_index:end_index]

        output_file = f"{output_prefix}_part_{i+1}_{n_splits}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(split_lines)

        print(f"Wrote {len(split_lines)} lines to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split a JSONL file into n evenly sized parts.')
    parser.add_argument('input_file', help='Path to the input JSONL file')
    parser.add_argument('n_splits', type=int, help='Number of splits')
    parser.add_argument('-o', '--output_prefix', default='split', help='Prefix for the output files')

    args = parser.parse_args()

    if args.n_splits < 1:
        print("Number of splits must be at least 1.")
        sys.exit(1)

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"The input file {args.input_file} does not exist.")
        sys.exit(1)

    split_jsonl_file(args.input_file, args.n_splits, args.output_prefix)

if __name__ == '__main__':
    main()
