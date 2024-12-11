#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "systolic_array_connection.h"
#include "aladdin/gem5/aladdin_sys_connection.h"
#include "fp16.h"
#include "gemm_simple_lib.h"

#define CACHELINE_SIZE 32

typedef struct {
    int m, k, n;
} MetaData;

#define MAX_DIM 16


void gemm_mock(float *inputs, float *weights, float *outputs, MetaData meta_data) {

    // Allocate padded matrices
    float16 *padded_inputs, *padded_weights, *padded_outputs;

    int input_dims[4] = { 1, meta_data.m, meta_data.k, 1 };
    int weight_dims[4] = { 1, meta_data.k, meta_data.n, 1 };
    // int output_dims[4] = { 1, meta_data.m, meta_data.n, 1 };
    int input_halo_pad[4] = { 0, 0, 0, 0 };

    int padded_input_dims[4] = { 1, MAX_DIM, MAX_DIM, 1 };
    int padded_weight_dims[4] = { 1, MAX_DIM, MAX_DIM, 1 };
    int padded_output_dims[4] = { 1, MAX_DIM, MAX_DIM, 1 };

    int input_size = 1, weight_size = 1, output_size = 1;
    for (int i = 0; i < 4; i++) {
        input_size *= padded_input_dims[i];
        weight_size *= padded_weight_dims[i];
        output_size *= padded_output_dims[i];
    }
    
    int err = posix_memalign((void**)&padded_inputs, CACHELINE_SIZE, sizeof(float16) * input_size);
    assert(err == 0 && "Failed to allocate memory!");
    err = posix_memalign((void**)&padded_weights, CACHELINE_SIZE, sizeof(float16) * weight_size);
    assert(err == 0 && "Failed to allocate memory!");
    err = posix_memalign((void**)&padded_outputs, CACHELINE_SIZE, sizeof(float16) * output_size);
    assert(err == 0 && "Failed to allocate memory!");

    // Copy input data with proper padding
    for (int i = 0; i < input_dims[1]; i++) {
        for (int j = 0; j < input_dims[2]; j++) {
            padded_inputs[i * padded_input_dims[2] + j] = fp16(inputs[i * input_dims[2] + j]);
        }
    }
    
    // Copy weight data with proper padding
    for (int i = 0; i < weight_dims[1]; i++) {
        for (int j = 0; j < weight_dims[2]; j++) {
            padded_weights[i * padded_weight_dims[2] + j] = fp16(weights[i * weight_dims[2] + j]);
        }
    }

    // // print inputs
    // for (int i = 0; i < padded_input_dims[1]; i++) {
    //     for (int j = 0; j < padded_input_dims[2]; ++j) {
    //         printf("inputs[%d]=%.2f ",i * padded_input_dims[2] + j, fp32(padded_inputs[i * padded_input_dims[2] + j]));
    //         // printf("%.2f ", inputs[i * padded_input_dims[2] + j]);
    //     }
    //     printf("\n");
    // }

    // // print weights
    // for (int i = 0; i < padded_input_dims[1]; i++) {
    //     for (int j = 0; j < padded_input_dims[2]; ++j) {
    //         printf("weights[%d]=%.2f ",i * padded_weight_dims[2] + j, fp32(padded_weights[i * padded_input_dims[2] + j]));
    //         // printf("%.2f ", inputs[i * padded_input_dims[2] + j]);
    //     }
    //     printf("\n");
    // }
    
    statAcceleratorUtil(input_dims, padded_input_dims, weight_dims, padded_weight_dims);
    storeArrayToAccelerator(padded_weights, padded_weight_dims);

    // Setup systolic array parameters
    systolic_array_params_t data;                   
    data.input_base_addr = &padded_inputs[0];
    data.weight_base_addr = &padded_weights[0];
    data.output_base_addr = &padded_outputs[0];

    memcpy(data.input_dims, padded_input_dims, sizeof(int) * 4);
    memcpy(data.weight_dims, padded_weight_dims, sizeof(int) * 4);
    memcpy(data.output_dims, padded_output_dims, sizeof(int) * 4);
    data.stride = 1;
    memcpy(data.input_halo_pad, input_halo_pad, sizeof(int) * 4);
    data.ifmap_start = 0;
    data.kern_start = 0;
    data.accum_results = false;
    data.read_inputs = true;
    data.read_weights = true;
    data.send_results = true;
    data.act_type = SYSTOLIC_RELU;
    int accelerator_id = 4;

    mapArrayToAccelerator(
        accelerator_id, "", data.input_base_addr, input_size * sizeof(float16));
    mapArrayToAccelerator(
        accelerator_id, "", data.weight_base_addr, weight_size * sizeof(float16));
    mapArrayToAccelerator(
        accelerator_id, "", data.output_base_addr, output_size * sizeof(float16));
    invokeSystolicArrayAndBlock(accelerator_id, data);    

    // Copy results back with unpadding
    for (int i = 0; i < meta_data.m; i++) {
        for (int j = 0; j < meta_data.n; j++) {
            outputs[i * meta_data.n + j] = fp32(padded_outputs[i * MAX_DIM + j]);
        }
    }

    // // print outputs
    // for (int i = 0; i < padded_output_dims[1]; i++) {
    //     for (int j = 0; j < padded_output_dims[2]; ++j) {
    //         printf("outputs[%d]=%.2f ",i * padded_output_dims[2] + j, fp32(padded_outputs[i * padded_output_dims[2] + j]));
    //         // printf("%.2f ", outputs[i * padded_output_dims[2] + j]);
    //     }
    //     printf("\n");
    // }

    // // print outputs
    // for (int i = 0; i < meta_data.m; i++) {
    //     for (int j = 0; j < meta_data.n; j++) {
    //         printf("outputs[%d]=%.2f ",i * meta_data.n + j, outputs[i * meta_data.n + j]);
    //     }
    //     printf("\n");
    // }

    free(padded_inputs);
    free(padded_weights);
    free(padded_outputs);
}
