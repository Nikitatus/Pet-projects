# Model weights

The two exported photo-to-art generator state dictionaries are stored here:

- `van_gogh_generator.pt`
- `monet_generator.pt`

Generate them from trusted training checkpoints with `export_generator.py`.
The training setup in `cyclegan.ipynb` uses artwork as domain A and photos as
domain B, so `generator_B` is the correct photo-to-art network.
