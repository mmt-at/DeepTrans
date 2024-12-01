from coder.testgenerator import TestGenerator
import subprocess
import os
import tempfile

example_code = """def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""


def test_testgenerator():
    generator = TestGenerator(model="deepseek-coder")
    unittest = generator.generate_tests(example_code)
    # assemble unittest with example code
    output = ""
    output += example_code
    output += unittest
    print(output)

    # save the output to a temporary python file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(output.encode())
        temp_file_path = temp_file.name

    # run the temporary python file
    process = subprocess.run(["python", temp_file_path], capture_output=True, text=True)
    print(process.stdout)
    print(process.stderr)
    if process.returncode != 0:
        raise Exception("Test failed")
    else:
        print("Test passed")

    # clean up the temporary file
    os.remove(temp_file_path)


test_testgenerator()
