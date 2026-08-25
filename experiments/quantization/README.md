# Model Quantization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nikitatus/Pet-projects/blob/main/experiments/quantization/quantization.ipynb)

Experiments with post-training quantization for LLMs, ranging from basic approaches to GPTQ.

## Contents

- `quantization.ipynb` — demo, explanations, implementations, and experiments.
- `kernel.cpp` and `kernel.cu` — custom INT8 and INT4 matrix-multiplication extension sources used by the notebook.

The notebook is designed to be run on a T4 GPU, for example on Google Colab.
