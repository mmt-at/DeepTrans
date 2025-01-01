typedef struct {
    float x, y, z;
} float3;
typedef struct {
    float x, y, z, w;
} float4;
// 添加 cpu_rasterizer_transformPoint4×4 函数的声明
float4 cpu_rasterizer_transformPoint4x4(const float3 p, const float* matrix);
float4 cpu_rasterizer_transformPoint4x4(const float3 p, const float* matrix) {
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}