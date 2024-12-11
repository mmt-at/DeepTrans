#include "gemm_simple_lib.h"

float global_utilization = 0;
int stat_num = 0;

#define CACHELINE_SIZE 32

void statAcceleratorUtil(int* input_dims, int* padded_input_dims, int* weight_dims, int* padded_weight_dims) {
    // 计算利用率
    float cur_util = 0.5;
    printf("utilization = %f\n", cur_util);

    // 统计利用率
    global_utilization += cur_util;
    stat_num++;
}

void printUtil() {
    printf("global utilization = %f\n", global_utilization / stat_num);
}

void storeArrayToAccelerator(float16* arr, int* dims) {
    // 临时数组存储转置后的矩阵
    int m = dims[1];
    int n = dims[2];
    float16* temp;
    int err = posix_memalign(
        (void**)&temp, CACHELINE_SIZE, sizeof(float16) * m * n);
    assert(err == 0 && "Failed to allocate memory!");

    // 填充临时数组为转置矩阵
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // 原始矩阵索引：i * n + j
            // 转置矩阵索引：j * m + i
            temp[j * m + i] = arr[i * n + j];
        }
    }

    // 将临时数组内容覆盖到原数组
    for (int i = 0; i < m * n; ++i) {
        arr[i] = temp[i];
    }

    // 交换dims维度
    dims[1] = n;
    dims[2] = m;

    free(temp);
    return;
}

float16 fp16(float fp32_data) {
    return fp16_ieee_from_fp32_value(fp32_data); 
}

float fp32(float16 fp16_data) { 
    return fp16_ieee_to_fp32_value(fp16_data); 
}