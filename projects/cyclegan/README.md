# CycleGAN Art Style Transfer

A Streamlit application that transforms one uploaded photo into Van Gogh and
Monet-style paintings together using separately trained CycleGAN generators.

**Live demo:** [CycleGAN Art Style Transfer](https://nikita-tkachuk-devopsml.hf.space/)

## Repository contents

- `app.py` — Streamlit upload UI, preprocessing, and two-style inference.
- `model.py` — the nine-block ResNet generator used by the notebook.
- `export_generator.py` — extracts a generator-only state dictionary from a
  trusted training checkpoint.
- `weights/` — the Van Gogh and Monet generator weights.
- `cyclegan.ipynb` — model implementation, training, and experiments.

## Trained weights

This project trains with artwork as domain A and photos as domain B. Therefore,
`generator_B` is the network used for photo-to-art inference. The exported Van
Gogh and Monet generator weights are included under `weights/`.

To replace the Van Gogh weights with a newer checkpoint, run:

```bash
python export_generator.py /path/to/van_gogh_checkpoint.pt \
  weights/van_gogh_generator.pt
```

To replace the Monet weights, export the separately trained `monet2photo`
checkpoint:

```bash
python export_generator.py /path/to/monet_checkpoint.pt \
  weights/monet_generator.pt
```

Only export checkpoints that you created or otherwise trust. The deployed app
loads the generator-only files with `weights_only=True`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The generator is fully convolutional. Uploaded images keep their aspect ratio,
are limited to a maximum side of 768 pixels, and are adjusted to dimensions
divisible by four for the two downsampling and upsampling stages.

## Deploy free on Streamlit Community Cloud

1. Create a GitHub repository containing the contents of this directory.
2. Make sure Git LFS is installed before committing the `.pt` weights:

   ```bash
   git lfs install
   git add .
   git commit -m "Add CycleGAN Streamlit app"
   git push
   ```

3. Sign in at `share.streamlit.io` with GitHub and create an app.
4. Select the repository and branch, then set the entrypoint to `app.py`.
5. Deploy the app. No secrets are required for this project.

The `.gitattributes` file configures `.pt` and `.pth` files for Git LFS. Both
model files must be present under `weights/` in the deployed repository.
