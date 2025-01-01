typedef struct {
    float x, y, z;
} float3;
// 添加 cpu_rasterizer_transformPoint4x3 函数的声明
float3 cpu_rasterizer_transformPoint4x3(const float3 p, const float* matrix);
float3 cpu_rasterizer_transformPoint4x3(const float3 p, const float* matrix) {
    float3 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8]  * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9]  * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
    return transformed;
}