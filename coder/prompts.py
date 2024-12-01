pure_system_prompt = """
You are a helpful assistant. You will follow the user's instructions to complete the task.
"""

coding_system_prompt = """
You are a helpful coding assistant. You will follow the user's instructions to complete the task.
"""
direct_replace_to_aladdin = """
Translate the following C code by embedding matrix multiplication calls and matrix-vector multiplication calls. The APIs are defined as:
```c
struct MetaData {{
    int m, n, k;
}};
void gemm_mock(const float* A, const float* B, float* C, MetaData meta_data);
```
The operation is equivalent to `C = A * B`, where `A` is an `m×k` matrix, `B` is a `k×n` matrix, and `C` is an `m×n` matrix.

```c
void gemv_mock(const float* A, const float* x, float* y, MetaData meta_data);
```
The operation is equivalent to `y = A * x`, where n = 1, `A` is an `m×k` column-major matrix, `x` is an `k×1` vector, and `y` is an `m×1` vector. Sometimes, `x`is not a `k×1` vector which is homogenized to a `k×1` vector by adding a 1 at the end.

The matrix multiplication operation is not always directly visible in the code. In such cases, you need to identify the matrix multiplication patterns and replace them with the corresponding `gemm_mock` or `gemv_mock` calls.

**Steps:**

1. **Function Separation:** Split the code by function boundaries, marking each function with ```code```.

2. **Code Fragment Analysis:** Within each function, split the code into fragments based on matrix multiplication semantics, marking each fragment with ```code```.

3. **Pattern Matching and Replacement:**
   - For code fragments containing matrix multiplication patterns, replace with `gemm_mock` calls and mark as `replace_gemm_fragments_{{number}}`.
   - For code fragments containing matrix-vector multiplication patterns, replace with `gemv_mock` calls and mark as `replace_gemv_fragments_{{number}}`.
   - For other code fragments, keep original code and mark as `function_fragments`.

4. **Output Format:** Output all code fragments sequentially using the following categories, ensuring the combined fragments exactly match the original function.

**Category Labels:**

- `include_fragments`: Include statements and non-function code
- `function_fragments`: Function body code without matrix multiplication
- `match_gemm_fragments_{{number}}`: Code fragments matching matrix multiplication pattern
- `replace_gemm_fragments_{{number}}`: Replacement code using gemm_mock
- `no_match_function`: Functions without matrix multiplication operations

Please prefix each code fragment with its corresponding label.

The followings are two examples to help you understand the task better.

**Example1:**

For input code:
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

Expected output format:

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

**Example2:**

For input code:
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void test3(int m, int n, int k) {{
    float MA[m * k] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    float VB[k * n] = {{1, 2, 3, 4}};
    float VC[m * n] = {{
        MA[0] * VB[0] + MA[4] * VB[2] + MA[8] * VB[3] + MA[12] * VB[4],
        MA[1] * VB[0] + MA[5] * VB[2] + MA[9] * VB[3] + MA[13] * VB[4],
        MA[2] * VB[0] + MA[6] * VB[2] + MA[10] * VB[3] + MA[14] *  VB[4],  
    }};

    return;
}}   

int main() {{
    int m = 3, n = 1, k = 4;
    test3(m, n, k);
    return 0;
}}
```

Expected output format:

# include_fragments
```c
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
```
# function_fragments
```c
void test3(int m, int n, int k) {{
    float MA[m * k] = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    float VB[k * n] = {{1, 2, 3, 4}};
```
# match_gemm_fragments_3
```c
    float VC[m * n] = {{
        MA[0] * VB[0] + MA[4] * VB[2] + MA[8] * VB[3] + MA[12] * VB[4],
        MA[1] * VB[0] + MA[5] * VB[2] + MA[9] * VB[3] + MA[13] * VB[4],
        MA[2] * VB[0] + MA[6] * VB[2] + MA[10] * VB[3] + MA[14] *  VB[4],  
    }};
```
- replace_gemm_fragments_3
```c
    MetaData meta_data;
    meta_data.m = m;
    meta_data.n = n;
    meta_data.k = k;
    gemv_mock(MA, VB, VC, meta_data);
```
# function_fragments
```c
    return;
}}
```
# no_match_function
```c
int main() {{
    int m = 3, n = 1, k = 4;
    test3(m, n, k);
    return 0;
}}
```


**Code to be transformed:**
{code_str}
"""
# 单文件内容拆分prompt
single_file_decompose_prompt = """Please analyze and decompose the following code into different sections:

The code should be decomposed into logical sections based on closure-like granularity. Each section should be marked with appropriate tags.

Here's a one-shot example:

Input code:

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
