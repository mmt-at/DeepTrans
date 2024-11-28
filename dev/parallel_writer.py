"""
This script loads a parallel dataset containing code samples in C++, Java, and Python.
It iterates over the samples in the training set and writes the content of each sample
to separate files for each programming language in a specified output directory.
"""

import datasets
import os
dataset2 = datasets.load_dataset("mistral0105/TransCoderParallel")

output_dir = "tmp/parallel"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# first iterate file
for sample in dataset2['train']:
    cpp_content = sample['cpp']
    java_content = sample['java']
    python_content = sample['python']
    file_id = sample['id']
    
    with open(f"{output_dir}/{file_id}.cpp", 'w') as cpp_file:
        cpp_file.write(cpp_content)
    
    with open(f"{output_dir}/{file_id}.java", 'w') as java_file:
        java_file.write(java_content)
    
    with open(f"{output_dir}/{file_id}.py", 'w') as python_file:
        python_file.write(python_content)
        
    # TODO: create a unittest generation for each language, use the skeleton in coder
    
    # example case exit
    break
