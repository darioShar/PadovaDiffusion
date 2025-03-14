# Install and train diffusion-like generative models

I recommend using an environment manager like [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html). 

Install the required packages using:

```bash
pip install -r requirements.txt
```

To verify that everything works well on your machine, just run my commands as is. The default MLP model is very small and any machine with a reasonable cpu will handle it. 

## Usage

The main script is `diffusion.py`, which supports two modes: **training** and **generation**.

### Training

Train the diffusion model:

```bash
python diffusion.py train --epochs 50 --checkpoint_interval 25 --batch_size 250 --learning_rate 2e-3 --dataset gaussian_mixture 
```

#### Arguments:

- `--epochs`: Total number of training epochs.
- `--checkpoint_interval`: Save a checkpoint every C epochs.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--dataset` : The dataset to train on. Either `gaussian_mixture`, a simple two-mixture of Gaussian, or `mnist`.
- `--checkpoint_dir`: *(Optional)* Directory to save checkpoints (default: `./checkpoints`). The file names will be `checkpoint_epoch_x`, where `x` is the epoch.

Each checkpoint saves the model state, optimizer state, and a list of epoch losses.

### Generation

Generate new images using a trained checkpoint:

```bash
python diffusion.py generate --checkpoint_path ./checkpoints/checkpoint_epoch_50.pth --num_images 1000 --reverse_steps 100  --output_path images/generated.png --dataset gaussian_mixture 
```

#### Arguments:

- `--checkpoint_path`: Path to the checkpoint file.
- `--num_images`: Number of images to generate.
- `--reverse_steps`: Number of reverse diffusion steps.
- `--output_path`: File path to save the generated image grid.
- `--dataset` : The dataset the model was trained on. Either `gaussian_mixture`, a simple two-mixture of Gaussian, or `mnist`.
- `--get_samples_history`: *(Optional)* If provided, the full generation history (each diffusion step) is saved as a series of images.

