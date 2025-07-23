import os
import json
from datasets import load_dataset

def convert_parquet_to_jsonl(parquet_file_path, jsonl_output_path):
    """
    Converts a Parquet file to a JSONL file using the Hugging Face datasets library.

    Args:
        parquet_file_path (str): Path to the input Parquet file.
        jsonl_output_path (str): Path to save the output JSONL file.
    """
    # Load the Parquet file as a dataset
    print(f"Loading Parquet file from: {parquet_file_path}")
    dataset = load_dataset("parquet", data_files=parquet_file_path)

    # Extract the first split (usually 'train' unless specified otherwise)
    dataset_split = dataset[list(dataset.keys())[0]]

    # Open the output JSONL file for writing
    print(f"Writing JSONL file to: {jsonl_output_path}")
    with open(jsonl_output_path, "w", encoding="utf-8") as jsonl_file:
        # Iterate over each row in the dataset and write it as a JSON object
        for row in dataset_split:
            jsonl_file.write(json.dumps(row) + "\n")

    print("Conversion completed successfully!")

if __name__ == "__main__":
    # Define input and output file paths
    parquet_file = "/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/aime_2025/data/train-00000-of-00001-243207c6c994e1bd.parquet"
    jsonl_file = "/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/aime_2025/test.jsonl"

    # Ensure the input Parquet file exists
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"The Parquet file '{parquet_file}' does not exist.")

    # Perform the conversion
    convert_parquet_to_jsonl(parquet_file, jsonl_file)