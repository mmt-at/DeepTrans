"""
This script loads a parallel dataset containing code samples in C++, Java, and Python.
It iterates over the samples, generates unit tests for each language, we validate the 
generated tests by compiling and running them.
"""

import datasets
import os
from coder.testgenerator import (
    TestGenerator,
    JavaTestGenerator,
    CppTestGenerator,
    PythonTestGenerator,
)
from util.config import logger, Language

dataset2 = datasets.load_dataset("mistral0105/TransCoderParallel")

output_dir = "tmp/parallel"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# first iterate file
for sample in dataset2["train"]:

    cpp_content = sample["cpp"]
    java_content = sample["java"]
    python_content = sample["python"]
    file_id = sample["id"]

    with open(f"{output_dir}/{file_id}.cpp", "w") as cpp_file:
        cpp_file.write(cpp_content)

    with open(f"{output_dir}/{file_id}.java", "w") as java_file:
        java_file.write(java_content)

    with open(f"{output_dir}/{file_id}.py", "w") as python_file:
        python_file.write(python_content)

    # TODO: create a unittest generation for each language, use the skeleton in coder
    # test_generator = TestGenerator(model="gpt-4o", temperature=0.3, src_lang=Language.CPP)
    # ut_cpp = test_generator.generate_tests(code_str=cpp_content) # the return code is an executable unittest code
    # print(ut_cpp)

    # generate unittest for cpp
    cpp_generator = CppTestGenerator(
        model="gpt-4o", temperature=0.3, src_lang=Language.CPP
    )
    ut_cpp = cpp_generator.generate_tests(
        code_str=cpp_content
    )  # the return code is an executable unittest code
    print(ut_cpp)
    with open(f"{output_dir}/{file_id}_test.cpp", "w") as cpp_test_file:
        cpp_test_file.write(ut_cpp)

    # TODO: consider to warp the following part into a class like CppTestRunner and put it somewhere
    import subprocess
    # compile the cpp test file
    try:
        ret = subprocess.run(
            [
                "g++",
                f"{output_dir}/{file_id}_test.cpp",
                "-o",
                f"{output_dir}/{file_id}_test",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ret.returncode != 0:
            logger.warning(
                f"Failed to compile the test file for {file_id}: {ret.stderr}"
            )
        # run the compiled test file
        ret = subprocess.run(
            [f"{output_dir}/{file_id}_test"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(ret.stdout.decode())  # the results we want
    except Exception as e:
        logger.warning(f"Failed to compile or run the test file for {file_id}: {e}")
        
    # generate unittest for java, 
    # TODO: make sure it use the same input as cpp
    java_generator = JavaTestGenerator(
        model="gpt-4o", temperature=0.3, src_lang=Language.JAVA
    )
    ut_java = java_generator.generate_tests(
        code_str=java_content
    )  # the return code is an executable unittest code
    print(ut_java)
    with open(f"{output_dir}/{file_id}_test.java", "w") as java_test_file:
        java_test_file.write(ut_java)
        
    # TODO: write a JavaTestRunner
    pass

    # generate unittest for python
    # TODO: make sure it use the same input as cpp
    python_generator = PythonTestGenerator(
        model="gpt-4o", temperature=0.3, src_lang=Language.PYTHON
    )
    ut_python = python_generator.generate_tests(
        code_str=python_content
    )  # the return code is an executable unittest code
    print(ut_python)
    with open(f"{output_dir}/{file_id}_test.py", "w") as python_test_file:
        python_test_file.write(ut_python)
    
    # TODO: write a PythonTestRunner
    pass

    # example case exit
    break
