# Model Quantization

Experiments with post-training quantization for LLMs, ranging from basic approaches to GPTQ.

## Contents

- `model_quantization.ipynb` — demo, explanations, implementations, and experiments.
- `kernel.cpp` and `kernel.cu` — custom INT8 and INT4 matrix-multiplication extension sources used by the notebook.

The notebook is designed to be run on T4 GPU, for example on Google Colab.
