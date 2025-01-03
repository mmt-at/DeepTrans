from util.config import logger
from coder.analyzer import LangDesc
from coder.codebase import CodeBase
from util.config import GPTModel, DeepseekModel, Language
import os

"""
TestGenerator is used to generate unit tests for a given code snippet.
To make sure code translation is tested properly, we need unittests for both src_lang and tar_lang.
We use the following rules to keep the unittests consistent
Input: code1
1. call translator to translate code1 from src_lang to tar_lang, get code2
2. call testgenerator to generate unittests for code1, get test1
#(deprecated, not consistent)3. call testgenerator to generate unittests for code2, get test2
3. call translator to translate test1 from src_lang to tar_lang, get test2
TODO: find a more consistent way to generate test1 and test2 together
4. run test1 and test2 to check if code1 and code2 are equivalent

this class is used to generate test1
TODO: consider using inherited TestGenerator for different languages
"""


class TestGenerator(CodeBase):

    def __init__(
        self,
        model=GPTModel.GPT35_TURBO,
        use_local=False,
        temperature=0.3,
        peft_model="",
        src_lang=Language.PYTHON,
    ):
        super().__init__(model, use_local, temperature, peft_model)
        self.src_lang = src_lang

    def generate_tests(self, code_str: str) -> str:
        logger.info(f"Generating tests for {self.src_lang.name} code")
        prompt = f"""
You are about to generate unittest for the following {self.src_lang.name} code.
Please follow the following rules:
1. Ensure the tests call the functions in the provided code.
2. Ensure the tests generate more than 10 test cases, except for trivial cases.
3. Ensure the tests are well-structured and formatted.
4. Ensure the tests are written in {self.src_lang.value}.
5. Ensure the tests are wrapped between '```test' and '```'.
6. You should assume you don't know the expected output, 
so all you need to do is in 3 steps:
<1>. synthesize input based on function parameters(or more), 
<2>. call the tested function with the input
<3>. print the output.
7. Please output the synthesized unittest code in the following format:
```test
<the unittest code>
```
An example of the unittest code is as follows:
###Example begin###
You are about to generate unit tests for the following python code.
Example input:
```python
def add(a, b):
    return a + b
```
Example output(you should only generate the following part):
```test
def test_add():
    # <1>.synthesize input cases
    a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    out_list = []
    # <2>.call the tested function with the synthesized input, record the output
    for i in range(10):
        a = a_list[i]
        b = b_list[i]
        out_list.append(add(a, b))
    # <3>.print the output
    print(out_list)
test_add()
```
###Example end###
The following is the actual input code to generate tests for:
```{self.src_lang.value}
{code_str}
```
"""
        resp = self.chat(prompt)
        logger.debug(f"Generated tests: {resp}")
        test_code = resp.split("```test")[1].split("```")[0]
        return test_code


class CppTestGenerator(TestGenerator):
    def __init__(
        self,
        model=GPTModel.GPT35_TURBO,
        use_local=False,
        temperature=0.3,
        peft_model="",
        src_lang=Language.CPP,
    ):
        super().__init__(model, use_local, temperature, peft_model, src_lang)

    def generate_tests(self, code_str) -> str:
        logger.info(f"Generating tests for {self.src_lang.name} code")
        # TODO: use relative path
        input_example = open(
            "../dev/one_shot/cpp_example.cpp"
        ).read()
        output_example = open(
            "../dev/one_shot/cpp_testgen.cpp"
        ).read()
        prompt = f"""
You are about to generate unittest for the following {self.src_lang.name} code.
Please follow the following rules:
1. Ensure the tests call the functions in the provided code.
2. Ensure the tests generate more than 10 test cases, except for trivial cases.
3. Ensure the tests are well-structured and formatted.
4. Ensure the tests are written in {self.src_lang.name}.
5. Ensure the tests are wrapped between '```test' and '```'.
6. You should assume you don't know the expected output, 
7. Please output the synthesized unittest code in the following format:
```test
<the unittest code>
```
In short, all you need to do is in 3 steps:
1. copy the source code here
2. write the test cases, to be specific, follow the following steps:
<1>. synthesize input based on function parameters(or more), 
<2>. call the tested function with the input
<3>. print the output.
3. write a main function to call the test cases
An example of the unittest code is as follows:
###Example begin###
You are about to generate unit tests for the following c++ code.
Example input:
```{self.src_lang.value}
{input_example}
```
Example output(you should only generate the following part):
```test
{output_example}
```
###Example end###
The following is the actual input code to generate tests for:
```{self.src_lang.value}
{code_str}
```
"""
        resp = self.chat(prompt)
        logger.debug(f"Generated tests: {resp}")
        test_code = resp.split("```test")[1].split("```")[0]
        return test_code


class JavaTestGenerator(TestGenerator):
    def __init__(
        self,
        model=GPTModel.GPT35_TURBO,
        use_local=False,
        temperature=0.3,
        peft_model="",
        src_lang=Language.JAVA,
    ):
        super().__init__(model, use_local, temperature, peft_model, src_lang)

    # TODO: specify one-shot example
    def generate_tests(self, code_str) -> str:
        return super().generate_tests(code_str)


class PythonTestGenerator(TestGenerator):
    def __init__(
        self,
        model=GPTModel.GPT35_TURBO,
        use_local=False,
        temperature=0.3,
        peft_model="",
        src_lang=Language.PYTHON,
    ):
        super().__init__(model, use_local, temperature, peft_model, src_lang)

    # TODO: specify one-shot example
    def generate_tests(self, code_str) -> str:
        return super().generate_tests(code_str)


class AladdinTestGenerator(TestGenerator):
    def __init__(
        self,
        model=DeepseekModel.CODER,
        use_local=False,
        temperature=0,
        peft_model="",
        src_lang=Language.C,
    ):
        super().__init__(model, use_local, temperature, peft_model, src_lang)

    def generate_tests(self, code_str) -> str:
        logger.info(f"Generating tests for {self.src_lang.name} code")

        input_example = open(
            "../dev/one_shot/c_example.c"
        ).read()
        output_example = open(
            "../dev/one_shot/c_testgen.c"
        ).read()
        prompt = f"""
You are about to generate unittest for the following {self.src_lang.name} code using a Given-When-Then approach.
Please follow these rules:
1. Ensure the tests call the functions in the provided code.
2. Include at least 10 test cases managed by loop, except for trivial cases.
3. Each test should follow the Given-When-Then framework:
   - Given: Set up any necessary conditions or inputs.
   - When: Perform the action (calling the function).
   - Then: Verify/print the output or outcome.
4. Wrap the result between '```test' and '```'.
5. Add the header file of the tested function, remove the function definition in the test code.
Output the synthesized unittest code here:
```test
<the unittest code>
```
An example of the unittest code is as follows:
###Example begin###
You are about to generate unit tests for the following c++ code.
Example input:
```{self.src_lang.value}
{input_example}
```
Example output(you should only generate the following part):
```test
{output_example}
```
###Example end###
The following is the actual input code to generate tests for:
```{self.src_lang.value}
{code_str}
```
"""
        resp = self.chat(prompt)
        logger.debug(f"Generated tests: {resp}")
        test_code = resp.split("```test")[1].split("```")[0]
        return test_code
    

if __name__ == "__main__":
    input = open(
        "../data/file/c/test/computeCov3D.c"
    ).read()
    AladdinTestGenerator().generate_tests(input)

# computeCov3D.c