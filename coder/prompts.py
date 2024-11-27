pure_system_prompt = """
You are a helpful assistant. You will follow the user's instructions to complete the task.
"""

coding_system_prompt = """
You are a helpful coding assistant. You will follow the user's instructions to complete the task.
"""
direct_replace_to_aladdin = """
将下面的C代码翻译成嵌入对gemm_mock调用的代码，其中gemm_mock的api定义为:
void gemm_mock(int m, int k, int n, const float* A, const float* B, float* C);
其运算语义等价于C = A * B，其中A是m行k列的矩阵，B是k行n列的矩阵，C是m行n列的矩阵。

首先将代码按照函数为边界进行拆分，函数之间的单独成为一段```code```。
然后请以矩阵乘语义为边界，将每个函数内部的代码拆分为多个片段，每个片段使用```code```标记。
请以此给出每一个拆分后的代码片段，拆分时需要将所有代码片段以此输出，并保证拆分后的代码片段合成回去之后和原函数一模一样，不需要多余的输出，以下是一个例子，对于文件内容为：
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int test(int m, int n, int k) {{
    float A[m * k] = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float B[k * n] = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float C[m * n];
    
    for (int i = 0; i < m; i++) {{
        for (int j = 0; j < n; j++) {{
            C[i * n + j] = 0;
            for (int k = 0; k < k; k++) {{
                C[i * n + j] += A[i * k + k] * B[k * n + j];
            }}
        }}
    }}

    // Calculate RGB
    float R = C[0] * 0.299f + C[1] * 0.587f + C[2] * 0.114f;
    float G = C[3] * 0.299f + C[4] * 0.587f + C[5] * 0.114f;
    float B = C[6] * 0.299f + C[7] * 0.587f + C[8] * 0.114f;

    printf("R: %f, G: %f, B: %f\n", R, G, B);
    return R + G + B;
}}

int main() {{
    int m = 3, n = 3, k = 3;
    return test(m, n, k);
}}
```
请输出以下结果:
# include_fragments
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
```
# function_fragments
```c
int test(int m, int n, int k) {{
    float MA[m * k] = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float MB[k * n] = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float MC[m * n];
```
# match_gemm_fragments_1
```c
    for (int i = 0; i < m; i++) {{
        for (int j = 0; j < n; j++) {{
            MC[i * n + j] = 0;
            for (int k = 0; k < k; k++) {{
                MC[i * n + j] += MA[i * k + k] * MB[k * n + j];
            }}
        }}
    }}
```
- replace_gemm_fragments_1
```c
    MetaData meta_data;
    meta_data.m = m;
    meta_data.n = n;
    meta_data.k = k;
    gemm_mock(MA, MB, MC, meta_data);
```
# function_fragments
```c
    // Calculate RGB
    float R = C[0] * 0.299f + C[1] * 0.587f + C[2] * 0.114f;
    float G = C[3] * 0.299f + C[4] * 0.587f + C[5] * 0.114f;
    float B = C[6] * 0.299f + C[7] * 0.587f + C[8] * 0.114f;

    printf("R: %f, G: %f, B: %f\n", R, G, B);
    return R + G + B;
}}
```
# no_match_function
```c
int main() {{
    int m = 3, n = 3, k = 3;
    return test(m, n, k);
}}
```
分为如下几类，include_fragments, function_fragments, match_gemm_fragments_{{1, 2, 3, ...}}, replace_gemm_fragments_{{1, 2, 3, ...}}, no_match_function。在代码前面打上标签
include_fragments表示include片段以及非函数体的代码片段，function_fragments表示因为检测到gemm而被切分的函数体代码片段，match_gemm_fragments_{{1, 2, 3, ...}}表示匹配到gemm的代码片段，replace_gemm_fragments_{{1, 2, 3, ...}}表示替换gemm的代码片段，no_match_function表示没有可替换的函数。
以下是待转换代码:
{code_str}
"""
better_replace_gemm_prompt = """
将下面的C代码翻译成嵌入对`gemm_mock`调用的代码，其中`gemm_mock`的API定义为：
```c
void gemm_mock(int m, int k, int n, const float* A, const float* B, float* C);
```
其运算语义等价于`C = A * B`，其中`A`是`m`行`k`列的矩阵，`B`是`k`行`n`列的矩阵，`C`是`m`行`n`列的矩阵。

**步骤：**

1. **函数拆分：** 将代码按照函数为边界进行拆分，每个函数单独成为一段，用```code```标记。

2. **代码片段拆分：** 在每个函数内部，以矩阵乘运算语义为边界，将代码拆分为多个片段，每个片段使用```code```标记。

3. **匹配与替换：**
   - 对于匹配到矩阵乘运算语义的代码片段，替换为调用`gemm_mock`的代码，并标记为`replace_gemm_fragments_{编号}`。
   - 对于未匹配到矩阵乘运算语义的代码片段，直接输出原代码，并标记为`function_fragments`。

4. **输出格式：** 按照以下分类输出每个代码片段，拆分时需要按顺序输出所有代码片段，并保证拆分后的代码片段合并后与原函数完全一致。

**分类标签：**

- `include_fragments`：include片段以及非函数体的代码片段。
- `function_fragments`：函数体中未被替换的代码片段。
- `match_gemm_fragments_{编号}`：匹配到矩阵乘运算语义的代码片段。
- `replace_gemm_fragments_{编号}`：替换后的`gemm_mock`调用代码片段。
- `no_match_function`：没有可替换代码的函数。

请在代码片段前加上对应的标签。

**示例：**

对于文件内容为：
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int test(int m, int n, int k) {
    float A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[k * n] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float C[m * n];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int t = 0; t < k; t++) {
                C[i * n + j] += A[i * k + t] * B[t * n + j];
            }
        }
    }

    printf("Result: %f\n", C[0]);
    return 0;
}

int main() {
    int m = 3, n = 3, k = 3;
    return test(m, n, k);
}
```

请输出以下结果：

```
#include_fragments
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
```

function_fragments
```c
int test(int m, int n, int k) {
    float A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[k * n] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float C[m * n];
```

match_gemm_fragments_1
```c
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int t = 0; t < k; t++) {
                C[i * n + j] += A[i * k + t] * B[t * n + j];
            }
        }
    }
```

replace_gemm_fragments_1
```c
    gemm_mock(m, k, n, A, B, C);
```

function_fragments
```c
    printf("Result: %f\n", C[0]);
    return 0;
}
```

no_match_function
```c
int main() {
    int m = 3, n = 3, k = 3;
    return test(m, n, k);
}
```

**以下是待转换的代码：**
{code_str}
"""

"""This file contains the prompts for the AI model to translate 
domain-specific applications into code with domain-specific operation supports without modification to compilers.
We currently target C code.
Each step should be accompanied with a 1-shot example to demonstrate the expected input and output.
"""

translator_system_prompt = """You are an expert performance engineer with experience in optimizing C code.
Your job is to translate an arbitrary C code with automatic vectorization, matrix multiplication, and parallelization optimizations.
You need to do this job with careful thinking first and optimize the code step by step.
All the code snippets you inpur or output should be marked between '```c' and '```' for alignment and formatting.
If you need to generate multiple output, return them in a list, I mean, return with '[' and ']', and separate them with ','."""

"""
input: C code snippet
(Natural language -> repository -> CUDA -> C) -> aladdin-C
Tree 1:
input->(analyze if valid C code)->(analyze current functions)->(analyze single function)->simple function
                                    ->multiple functions->inputs         ->extract functions->inputs
                                        + function relationships                + function relationships
Tree 2:
simple function->(analyze)->(transform)->(unittest)->optimized function
Tree 3:
multiple optimized functions->(combine)->(end2end test)->optimized code snippet                                  
output: optimized code snippet
"""

analyzer_prompt_step1 = """#Step 1:
based on the input code snippet, you need to analyze whether the input is a valid C code, with all the necessary libraries and functions provided.
Note that the input code snippet should be at least a complete function or a set of functions that can be compiled and run.
You should return with 'YES' to indicate it is valid, or 'NO' to indicate it is invalid. You can also provide additional information if needed."""
analyzer_prompt_step2 = """#Step 2:
based on the input C code snippet, you need to analyze the functions, global values and their relationships.
If it contains only one function, return with the original input code snippet,
otherwise, you need to split the functions"""

# Prompt for analyzing a single function
analyzer_prompt_single_function = """#Single Function Analysis:
Analyze the given single function to understand its purpose, inputs, outputs, and any dependencies it might have.
Provide a detailed breakdown of the function's logic and any potential areas for optimization."""

# Prompt for analyzing multiple functions
analyzer_prompt_multiple_functions = """#Multiple Functions Analysis:
Analyze the given set of functions to understand their individual purposes, inputs, outputs, and interdependencies.
Provide a detailed breakdown of each function's logic, how they interact with each other, and any potential areas for optimization."""

# Prompt for extracting functions and their relationships
extractor_prompt_functions_relationships = """#Extract Functions and Relationships:
Identify and extract all functions from the input code snippet, along with their relationships.
Provide a clear mapping of function calls, shared variables, and any other dependencies."""

# Prompt for transforming a simple function
transformer_prompt_simple_function = """#Transform Simple Function:
Transform the analyzed simple function to optimize its performance.
Consider techniques such as loop unrolling, inlining, and other micro-optimizations.
Ensure the transformed function maintains the same functionality."""

# Prompt for unit testing an optimized function
unittest_prompt_optimized_function = """#Unit Test Optimized Function:
Create unit tests for the optimized function to ensure it behaves as expected.
Include tests for edge cases and typical usage scenarios."""

# Prompt for combining multiple optimized functions
combiner_prompt_multiple_optimized_functions = """#Combine Multiple Optimized Functions:
Combine the optimized functions into a cohesive code snippet.
Ensure that the combined code maintains the original functionality and is optimized for performance."""

# Prompt for end-to-end testing of the optimized code snippet
end2end_test_prompt_optimized_code = """#End-to-End Test Optimized Code Snippet:
Perform an end-to-end test on the combined optimized code snippet.
Verify that the code snippet compiles, runs correctly, and meets performance expectations."""
