#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "fp16.h"

typedef uint16_t float16;

extern float global_utilization;
extern int stat_num;

extern float16 fp16(float fp32_data);
extern float fp32(float16 fp16_data);

void statAcceleratorUtil(int* init_input_dims, int* pad_intput_dims, int* init_weight_dims, int* pad_weight_dims);

void printUtil();

// 这个函数在设置systolic_array_params_t之前必须使用
void storeArrayToAccelerator(float16* arr, int* dims);