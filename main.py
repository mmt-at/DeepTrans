from coder.translator import CUDA2CTranslator
from tools.compiler import CompilerCaller
if __name__ == "__main__":
    compiler = CompilerCaller("C")
    import os
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    repo_name = "test"
    target_func_name = "computeCov3D"
    # build_dir = compiler.build(position=data_folder, level="repo", lang="c", name=repo_name, target_func_name=target_func_name, inplace=False)
    # compiler.run_executable(build_dir, function_name=target_func_name)
    compiler.lowering(position=data_folder, level="repo", lang="c", name=repo_name, target_func_name=target_func_name, inplace=False)
    exit(0)
    translator = CUDA2CTranslator(model="deepseek-coder")
    translator.translate("""// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	// S = 	| 1.0  0.0  0.0 |
	// 		| 0.0  1.0  0.0 |
	// 		| 0.0  0.0  1.0 |
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}
""")