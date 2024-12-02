#include "math.h"

typedef struct {
    float x, y, z;
} float3;

// 添加函数computeCov2D的声明
float3 computeCov2D(const float3 mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D, const float *viewmatrix);

float3 computeCov2D(const float3 mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D, const float *viewmatrix)
{
    float3 t = cpu_rasterizer_transformPoint4x3(mean, viewmatrix);

    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = fmin(limx, fmax(-limx, txtz)) * t.z;
    t.y = fmin(limy, fmax(-limy, tytz)) * t.z;

    float J[9] = {
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f};

    float W[9] = {
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]};

    float T[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            T[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                T[i * 3 + j] += W[i * 3 + k] * J[k * 3 + j];
            }
        }
    }

    float Vrk[9] = {
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]};

    float TT[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            TT[i * 3 + j] = T[j * 3 + i];
        }
    }

    float Vrk_T[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Vrk_T[i * 3 + j] = Vrk[j * 3 + i];
        }
    }

    float temp[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            temp[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                temp[i * 3 + j] += TT[i * 3 + k] * Vrk_T[k * 3 + j];
            }
        }
    }

    float cov[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cov[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                cov[i * 3 + j] += temp[i * 3 + k] * T[k * 3 + j];
            }
        }
    }

    cov[0] += 0.3f;
    cov[4] += 0.3f;

    float3 result = {cov[0], cov[1], cov[4]};
    return result;
}