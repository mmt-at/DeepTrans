typedef struct {
    float x, y, z;
} float3;

typedef struct {
    float x, y, z, w;
} float4;

// 添加 computeCov3D 函数的声明
void computeCov3D(float3 scale, float mod, float4 rot, float* cov3D);
