class TemplateFiller:
    _id = 0
    @staticmethod
    def aladdin_original_conv_template(data_type: str = "float16",
                                       input_dims: list[int] = [1, 16, 16, 8],
                                       weight_dims: list[int] = [16, 3, 3, 8],
                                       output_dims: list[int] = [1, 8, 8, 16],
                                       input_halo_pad: list[int] = [1, 1, 1, 1],
                                       stride: int = 2,
                                       accelerator_id: int = 4,
                                       func_name: str = None,
                                       check_code: str = ""):
        if func_name is None:
            func_name = f"aladdin_original_conv_{TemplateFiller._id}"
            TemplateFiller._id += 1
        header = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "systolic_array_connection.h"
#include "aladdin/gem5/aladdin_sys_connection.h"
#include "fp16.h"
"""
        body = f"""
void {func_name}({data_type} *inputs, {data_type} *weights, {data_type} *outputs) {{
    int input_dims[{len(input_dims)}] = {input_dims};
    int weight_dims[{len(weight_dims)}] = {weight_dims};
    int output_dims[{len(output_dims)}] = {output_dims};
    int input_halo_pad[{len(input_halo_pad)}] = {input_halo_pad};   
    int err = posix_memalign(
        (void**)&inputs, CACHELINE_SIZE, sizeof({data_type}) * input_size);
    assert(err == 0 && "Failed to allocate memory!");
    err = posix_memalign(
        (void**)&weights, CACHELINE_SIZE, sizeof({data_type}) * weight_size);
    assert(err == 0 && "Failed to allocate memory!");
    err = posix_memalign(
        (void**)&outputs, CACHELINE_SIZE, sizeof({data_type}) * output_size);
    assert(err == 0 && "Failed to allocate memory!");

    for (int i = 0; i < input_size; i++) {{
        inputs[i] = fp16(inputs[i]);
    }}
    for (int i = 0; i < weight_size; i++) {{
        weights[i] = fp16(weights[i]);
    }}

    // construct systolic array parameters, a package of input/weight/output data
    systolic_array_params_t data;
    data.input_base_addr = &inputs[0];
    data.weight_base_addr = &weights[0];
    data.output_base_addr = &outputs[0];
    memcpy(data.input_dims, input_dims, sizeof(int) * 4);
    memcpy(data.weight_dims, weight_dims, sizeof(int) * 4);
    memcpy(data.output_dims, output_dims, sizeof(int) * 4);
    data.stride = {stride};
    memcpy(data.input_halo_pad, input_halo_pad, sizeof(int) * 4);
    data.ifmap_start = 0;
    data.kern_start = 0;
    data.accum_results = false;
    data.read_inputs = true;
    data.read_weights = true;
    data.send_results = true;
    data.act_type = SYSTOLIC_RELU;
    mapArrayToAccelerator(
        {accelerator_id}, "", data.input_base_addr, input_size * sizeof({data_type}));
    mapArrayToAccelerator(
        {accelerator_id}, "", data.weight_base_addr, weight_size * sizeof({data_type}));
    mapArrayToAccelerator(
        {accelerator_id}, "", data.output_base_addr, output_size * sizeof({data_type}));
    invokeSystolicArrayAndBlock({accelerator_id}, data);
    {check_code}
}}
"""
        return header, body
    
    
    @staticmethod
    def aladdin_simple_gemm_template(data_type: str = "float16",
                                     input_dims: list[int] = [16, 16],
                                     weight_dims: list[int] = [16, 16],
                                     func_name: str = None,
                                     check_code: str = ""):
        if func_name is None:
            func_name = f"aladdin_simple_gemm_{TemplateFiller._id}"
            TemplateFiller._id += 1
        header = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "systolic_array_connection.h"
#include "aladdin/gem5/aladdin_sys_connection.h"
#include "fp16.h"

#define CACHELINE_SIZE 32

typedef uint16_t float16;

float16 fp16(float fp32_data) { return fp16_ieee_from_fp32_value(fp32_data); }
float fp32(float16 fp16_data) { return fp16_ieee_to_fp32_value(fp16_data); }

struct accel {
    void* input_base_addr;
    int input_size;
    void* weight_base_addr;
    int weight_size;
    void* output_base_addr;
    int output_size;
};

struct accel data;

struct MetaData {
    int m, k, n;
};

"""
        body = f"""
void {func_name}(float *inputs, float *weights, float *outputs, MetaData meta_data) {{
    for (int i = 0; i < meta_data.m * meta_data.k; i++) {{
        inputs[i] = fp16(inputs[i]);
    }}
    for (int i = 0; i < meta_data.k * meta_data.n; i++) {{
        weights[i] = fp16(weights[i]);
    }}
    data.input_base_addr = inputs;
    data.weight_base_addr = weights;
    data.output_base_addr = outputs;
    data.input_size = meta_data.m * meta_data.k;
    data.weight_size = meta_data.k * meta_data.n;
    data.output_size = meta_data.m * meta_data.n;
    mapArrayToAccelerator(
        data.input_base_addr, data.input_size * sizeof(float16));
    mapArrayToAccelerator(
        data.weight_base_addr, data.weight_size * sizeof(float16));
    mapArrayToAccelerator(
        data.output_base_addr, data.output_size * sizeof(float16));
    invokeSystolicArrayAndBlock(meta_data);
    for (int i = 0; i < data.output_size; i++) {{
        outputs[i] = fp32(((float*)data.output_base_addr)[i]);
    }}
    {check_code}
}}
"""
        return header, body
    
    def aladdin_mock_gemm_template(data_type: str = "float16",
                                   input_dims: list[int] = [16, 16],
                                   weight_dims: list[int] = [16, 16]):
        header, body = TemplateFiller.aladdin_simple_gemm_template(data_type, input_dims, weight_dims, "gemm_mock", "")
        header = header.replace("#include \"systolic_array_connection.h\"\n#include \"aladdin/gem5/aladdin_sys_connection.h\"\n#include \"fp16.h\"\n\n#define CACHELINE_SIZE 32", "")
        header = header.replace("typedef uint16_t float16;", "typedef float float16;")
        header = header.replace("float16 fp16(float fp32_data) { return fp16_ieee_from_fp32_value(fp32_data); }", "float16 fp16(float fp32_data) { return fp32_data; }")
        header = header.replace("float fp32(float16 fp16_data) { return fp16_ieee_to_fp32_value(fp16_data); }", "float fp32(float16 fp16_data) { return fp16_data; }")
        header = header + """
void mapArrayToAccelerator(void *base_addr, size_t size) {
    return;
}

void invokeSystolicArrayAndBlock(MetaData meta_data) {
    for (int i = 0; i < meta_data.m; i++) {
        for (int j = 0; j < meta_data.n; j++) {
            ((float*)data.output_base_addr)[i * meta_data.n + j] = 0;
            for (int k = 0; k < meta_data.k; k++) {
                ((float*)data.output_base_addr)[i * meta_data.n + j] += ((float*)data.input_base_addr)[i * meta_data.k + k] * ((float*)data.weight_base_addr)[k * meta_data.n + j];
            }
        }
    }
}
"""
        return header, body


    @staticmethod
    def aladdin_mock_matrix_vector_template(data_type: str = "float16",
                                       input_dims: list[int] = [16, 16],
                                       weight_dims: list[int] = [16],
                                       func_name: str = None,
                                       check_code: str = ""):
        if func_name is None:
            func_name = f"aladdin_matrix_vector_{TemplateFiller._id}"
            TemplateFiller._id += 1
        header = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef float float16;

float16 fp16(float fp32_data) { return fp32_data; }
float fp32(float16 fp16_data) { return fp16_data; }

struct accel {
    void* input_base_addr;
    int input_size;
    void* weight_base_addr;
    int weight_size;
    void* output_base_addr;
    int output_size;
};

struct accel data;

struct MetaData {
    int m, n;
};

void mapArrayToAccelerator(void *base_addr, size_t size) {
    return;
}

void invokeSystolicArrayAndBlock(MetaData meta_data) {
    for (int i = 0; i < meta_data.m; i++) {
        ((float*)data.output_base_addr)[i] = 0;
        for (int j = 0; j < meta_data.n; j++) {
            ((float*)data.output_base_addr)[i] += ((float*)data.input_base_addr)[i * meta_data.n + j] * ((float*)data.weight_base_addr)[j];
        }
    }
}

"""
        body = f"""
void {func_name}(float *inputs, float *weights, float *outputs, MetaData meta_data) {{
    for (int i = 0; i < meta_data.m * meta_data.k; i++) {{
        inputs[i] = fp16(inputs[i]);
    }}
    for (int i = 0; i < meta_data.n; i++) {{
        weights[i] = fp16(weights[i]);
    }}
    data.input_base_addr = inputs;
    data.weight_base_addr = weights;
    data.output_base_addr = outputs;
    data.input_size = meta_data.m * meta_data.n;
    data.weight_size = meta_data.n;
    data.output_size = meta_data.m;
    mapArrayToAccelerator(
        data.input_base_addr, data.input_size * sizeof(float16));
    mapArrayToAccelerator(
        data.weight_base_addr, data.weight_size * sizeof(float16));
    mapArrayToAccelerator(
        data.output_base_addr, data.output_size * sizeof(float16));
    invokeSystolicArrayAndBlock(meta_data);
    for (int i = 0; i < data.output_size; i++) {{
        outputs[i] = fp32(((float*)data.output_base_addr)[i]);
    }}
    {check_code}
}}
"""
        return header, body
    
    def aladdin_mock_gemm_to_file(file_path: str = None):
        header, body = TemplateFiller.aladdin_mock_gemm_template()
        if file_path is None:
            import os
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output", "aladdin_mock_gemm.h")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(header)
            f.write(body)

if __name__ == "__main__":
    TemplateFiller.aladdin_mock_gemm_to_file()