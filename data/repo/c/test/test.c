#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "computeCov3D.h"

void test_computeCov3D() {
    float3 scale = {1.0f, 2.0f, 3.0f};
    float mod = 1.0f;
    float4 rot = {4.0f, 5.0f, 6.0f, 7.0f};
    float cov3D[6] = {0.0f};

    computeCov3D(scale, mod, rot, cov3D);

    printf("scale: %f, %f, %f\n", scale.x, scale.y, scale.z);
    printf("mod: %f\n", mod);
    printf("rot: %f, %f, %f, %f\n", rot.x, rot.y, rot.z, rot.w);

    printf("cov3D: %f, %f, %f, %f, %f, %f\n", cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
    fflush(stdout);

    // 这里我们假设cov3D的期望值是已知的
    float expected_cov3D[6] = {153941.0f, 24772.0f, -130236.0f, 117316.0f, -118276.0f, 193757.0f};

    for (int i = 0; i < 6; i++) {
        assert(fabs(cov3D[i] - expected_cov3D[i]) < 1e-6);
    }

    printf("test_computeCov3D passed.\n");
}

int main() {
    test_computeCov3D();
    return 0;
}