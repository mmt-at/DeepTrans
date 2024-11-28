"""
This script converts the jsonl file containing parallel samples into a Hugging Face dataset.
"""

import json
import datasets

def parse_transcoder_jsonl(file_path):
    data = []
    # Open and read the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    # Convert the data to dataset format
    dataset = datasets.Dataset.from_dict({
        'id': [item['id'] for item in data],
        'cpp': [item['cpp'] for item in data],
        'java': [item['java'] for item in data],
        'python': [item['python'] for item in data]
    })
    return dataset

# Parse the JSONL file and create a dataset
dataset = parse_transcoder_jsonl('data/single_file/parallel/parallel_samples.jsonl')

# Print dataset information
print("Number of samples:", len(dataset))
print("Column names:", dataset.column_names)
# print("First sample:", dataset[0])

# Push the dataset to Hugging Face Hub
dataset.push_to_hub("mistral0105/TransCoderParallel")