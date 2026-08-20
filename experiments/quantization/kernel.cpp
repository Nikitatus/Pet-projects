#include <torch/extension.h>

torch::Tensor int4MatmulCUDA(const torch::Tensor& lhs,
                            const torch::Tensor& rhs);
torch::Tensor int8MatmulCUDA(const torch::Tensor& lhs,
                            const torch::Tensor& rhs);

namespace {

void validateInputs(const char* operation, const torch::Tensor& lhs,
                    const torch::Tensor& rhs) {
  torch::checkAllContiguous(operation,
                            {{lhs, "A", 0}, {rhs, "B", 1}});
  torch::checkDeviceType(operation, {lhs, rhs}, at::DeviceType::CUDA);
}

}  // namespace

torch::Tensor int4Matmul(const torch::Tensor& lhs,
                        const torch::Tensor& rhs) {
  validateInputs("int4Matmul", lhs, rhs);
  return int4MatmulCUDA(lhs, rhs);
}

torch::Tensor int8Matmul(const torch::Tensor& lhs,
                        const torch::Tensor& rhs) {
  validateInputs("int8Matmul", lhs, rhs);
  return int8MatmulCUDA(lhs, rhs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("int4_matmul", &int4Matmul, "int4 matmul (CUDA)");
  module.def("int8_matmul", &int8Matmul, "int8 matmul (CUDA)");
}
