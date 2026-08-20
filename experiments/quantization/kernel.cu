#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>

torch::Tensor int4MatmulCUDA(const torch::Tensor& lhs,
                            const torch::Tensor& rhs) {
  torch::checkAllSameGPU("int4Matmul",
                         {{lhs, "A", 0}, {rhs, "B", 1}});

  const auto rows = lhs.size(0);
  const auto columns = rhs.size(0);
  const auto inner = lhs.size(1) * 2;
  auto result = torch::empty(
      {rows, columns}, torch::dtype(torch::kInt32).device(lhs.device()));

  using Int4Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,
      cutlass::layout::RowMajor,
      cutlass::int4b_t,
      cutlass::layout::ColumnMajor,
      int32_t,
      cutlass::layout::RowMajor,
      int32_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75>;
  using ProblemSize = cutlass::gemm::GemmCoord;

  const ProblemSize size(static_cast<ProblemSize::Index>(rows),
                         static_cast<ProblemSize::Index>(columns),
                         static_cast<ProblemSize::Index>(inner));
  typename Int4Gemm::Arguments args{
      size,
      {reinterpret_cast<cutlass::int4b_t*>(lhs.data_ptr<uint8_t>()), inner},
      {reinterpret_cast<cutlass::int4b_t*>(rhs.data_ptr<uint8_t>()), inner},
      {result.data_ptr<int32_t>(), columns},
      {result.data_ptr<int32_t>(), columns},
      {1, 0}};

  Int4Gemm runGemm;
  const auto status = runGemm(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return result;
}

torch::Tensor int8MatmulCUDA(const torch::Tensor& lhs,
                            const torch::Tensor& rhs) {
  torch::checkAllSameGPU("int8Matmul",
                         {{lhs, "A", 0}, {rhs, "B", 1}});

  const auto rows = lhs.size(0);
  const auto columns = rhs.size(0);
  const auto inner = lhs.size(1);
  auto result = torch::empty(
      {rows, columns}, torch::dtype(torch::kInt32).device(lhs.device()));

  using Int8Gemm = cutlass::gemm::device::Gemm<
      int8_t,
      cutlass::layout::RowMajor,
      int8_t,
      cutlass::layout::ColumnMajor,
      int32_t,
      cutlass::layout::RowMajor,
      int32_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75>;
  using ProblemSize = cutlass::gemm::GemmCoord;

  const ProblemSize size(static_cast<ProblemSize::Index>(rows),
                         static_cast<ProblemSize::Index>(columns),
                         static_cast<ProblemSize::Index>(inner));
  typename Int8Gemm::Arguments args{
      size,
      {lhs.data_ptr<int8_t>(), inner},
      {rhs.data_ptr<int8_t>(), inner},
      {result.data_ptr<int32_t>(), columns},
      {result.data_ptr<int32_t>(), columns},
      {1, 0}};

  Int8Gemm runGemm;
  const auto status = runGemm(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return result;
}
