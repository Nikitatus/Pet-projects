# Image Upscaling Test Task

Here are instructions to setup inference and evaluation of the models for image super-resolution.

Results dataset: [nikita-tkachuk/upscaler-results](https://huggingface.co/datasets/nikita-tkachuk/upscaler-results)

## Setup Instructions

### 1. Generating Upscaled Images (Inference)

#### AuraSR
To run the provided AuraSR inference logic:

```bash
python evaluation/aura_sr_test.py \
    --orig data/original \
    --out output/AuraSR \
    --device cuda
```
 The script captures latency and VRAM metrics and resamples 4x output to 2x and 3x factors, as the original model outputs 4x upscaled images only.

#### HAT (via Docker)
No specific Python scripts are provided in this repository for HAT. Inference was carried out using a pre-configured Docker image. You can pull the image and run testing using only the command which is provided in the original [HAT GitHub repository](https://github.com/XPixelGroup/HAT):
```bash
python hat/test.py -opt options/test/HAT_SRx4.yml
```

You can change all the parameters of the model by modifying the `options/test/HAT_SRx4.yml` file.

Here is the instruction on how to run the model in docker:
```bash
# Pull the docker image
docker pull nikitatus/hat:v1

# Run inference via docker
docker run --gpus all \
  -v $(pwd)/data:/upscaler/data \
  -v $(pwd)/output:/upscaler/output \
  -w /upscaler/HAT \
  nikitatus/hat:v1 \
  python3 test.py -opt options/test/HAT_SRx4.yml # 
```
To change the upscale, switch to different config file like options/test/HAT_SRx2.yml. To change the input/output paths, modify the config 'yml' file.

No extra steps needed!

#### TSD-SR (via Docker)
Similarly, TSD-SR inference: 

```bash
# Pull the docker image
docker pull nikitatus/tsd-sd-upscaler:v1

# Run inference via docker
docker run --gpus all \
  -v $(pwd)/data:/upscaler/data \
  -v $(pwd)/output:/upscaler/output \
  -w /upscaler/TSD-SR \
  nikitatus/tsd-sd-upscaler:v1 \
  python3 test/test_tsdsr.py \
    --pretrained_model_name_or_path ./sd3-medium \
    -i imgs/test \ # this is only thing you need to change
    -o outputs/test \ # and this if you want
    --lora_dir checkpoint/tsdsr \
    --embedding_dir dataset/default
```

Performance logs and metrics for these Docker-based models were captured directly from the Docker container output.

Link to the original repository: https://github.com/XPixelGroup/TSD-SR

### 2. Computing Image Quality Metrics

To compute quantitative evaluation metrics comparing the model outputs back to the original inputs (e.g., for models running locally like AuraSR):

```bash
python evaluation/run_evaluation.py \
    --orig data/ground_truth_1k \
    --upscaled output/AuraSR \
    --metrics psnr ssim lpips \
    --device cuda
```

**Options for `--metrics`:**
You can specify one or more metrics: `psnr`, `ssim`, `lpips`, `fid`. By default, all 4 are computed.

The same metrics can be computed for the other models by changing the `--upscaled` argument to the path of the corresponding model's output directory.


> **Note on Directory Scheme:** The directory structure slightly departs from the initially requested format. Specifically, there is no `metrics_summary.csv` and no custom `.py` inference scripts for the HAT and TSD-SR models. This is because HAT and TSD-SR were evaluated using pre-configured Docker images from Docker Hub by following the instructions in their respective GitHub repositories. Instead of a single CSV, metrics and logs are provided individually after the inference of each model. To check the metrics and logs for each model, you can refer to the output directory of each model or you can find this in the report.
