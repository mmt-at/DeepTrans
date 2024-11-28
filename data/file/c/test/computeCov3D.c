typedef struct {
    float x, y, z;
} float3;

typedef struct {
    float x, y, z, w;
} float4;

// 添加 computeCov3D 函数的声明
void computeCov3D(float3 scale, float mod, float4 rot, float* cov3D);

void computeCov3D(float3 scale, float mod, float4 rot, float *cov3D)
{
    float S[9] = {
        mod * scale.x, 0.0f, 0.0f,
        0.0f, mod * scale.y, 0.0f,
        0.0f, 0.0f, mod * scale.z};

    // float qx = rot.x;
    // float qy = rot.y;
    // float qz = rot.z;
    // float qw = rot.w;
    float qr = rot.x;
	float qx = rot.y;
	float qy = rot.z;
	float qz = rot.w;

    float R[9] = {
        1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - qr * qz), 2.f * (qx * qz + qr * qy),
        2.f * (qx * qy + qr * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - qr * qx),
        2.f * (qx * qz - qr * qy), 2.f * (qy * qz + qr * qx), 1.f - 2.f * (qx * qx + qy * qy)};

    float M[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            M[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                M[i * 3 + j] += R[i * 3 + k] * S[k * 3 + j];
            }
        }
    }

    float MT[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            MT[i * 3 + j] = M[j * 3 + i];
        }
    }

    float Sigma[9];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Sigma[i * 3 + j] = 0.0f;
            for (int k = 0; k < 3; k++)
            {
                Sigma[i * 3 + j] += M[i * 3 + k] * MT[k * 3 + j];
            }
        }
    }

    
    cov3D[0] = Sigma[0];
    cov3D[1] = Sigma[1];
    cov3D[2] = Sigma[2];
    cov3D[3] = Sigma[4];
    cov3D[4] = Sigma[5];
    cov3D[5] = Sigma[8];
}
