#!/usr/bin/env python
"""
Generative Diffusion Model on simple datasets

This script demonstrates:
  - Creating the DiffusionProcess object, U-Net model, and optimizer.
  - Loading MNIST with basic data augmentation (random horizontal flipping).
  - Or loading a simple Gaussian mixture dataset.
  - Training the model for N epochs with checkpointing every C epochs.
  - A generation procedure that loads a checkpoint and generates N images using T reverse steps.
  - Saving the generated images (and optionally a history of the generation process).

Usage:
  Training:
    python diffusion.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-3

  Generation:
    python diffusion.py generate --checkpoint_path ./checkpoints/checkpoint_epoch_20.pth \
      --num_images 16 --reverse_steps 100 --output_path generated.png
"""

import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

import model.unet as unet
import model.SimpleModel as SimpleModel

import matplotlib.pyplot as plt

#############################################
# Helper function 
#############################################

def match_last_dims(data, size):
    """
    Repeat a 1D tensor so that its last dimensions [1:] match `size[1:]`.
    Useful for working with batched data.
    """
    assert len(data.size()) == 1, "Data must be 1-dimensional (one value per batch)"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

class DiffusionProcess:
    def __init__(self, device=None, T=1.0, process_type='VP', schedule='linear', **kwargs):
        """
        process_type: 'VP' (variance preserving) or 'VE' (variance exploding)
        schedule: for VP, choose 'linear' or 'cosine'
        T: time horizon (used to sample t ~ Uniform(0,T)); the neural net always receives normalized time in [0,1]
        """
        self.device = 'cpu' if device is None else device
        self.T = T
        self.process_type = process_type
        self.schedule = schedule
        
        if process_type == 'VP':
            if schedule == 'linear':
                self.beta_min = kwargs.get('beta_min', 0.1)
                self.beta_max = kwargs.get('beta_max', 20.0)
            elif schedule == 'cosine':
                # s is a small offset (default 0.008) as in Nichol & Dhariwal’s cosine schedule.
                self.s = kwargs.get('s', 0.008)
            else:
                raise ValueError("Unknown VP schedule type")
        elif process_type == 'VE':
            self.sigma_min = kwargs.get('sigma_min', 0.01)
            self.sigma_max = kwargs.get('sigma_max', 50.0)
        else:
            raise ValueError("Unknown process type")
    
    def alpha_bar(self, t_norm):
        """
        For VP processes, returns the cumulative product (or survival probability) at normalized time t_norm.
        For linear: ᾱ(t) = exp( - [β_min t + 0.5 (β_max - β_min)t^2] )
        For cosine: ᾱ(t) = cos( ((t + s)/(1+s))*(π/2) )^2
        """
        if self.process_type == 'VP':
            if self.schedule == 'linear':
                integrated_beta = self.beta_min * t_norm + 0.5 * (self.beta_max - self.beta_min) * t_norm**2
                return torch.exp(-integrated_beta)
            elif self.schedule == 'cosine':
                return torch.cos((t_norm + self.s) / (1 + self.s) * (math.pi / 2))**2
        else:
            return None

    def sigma_fn(self, t_norm):
        """
        For VE processes, returns the noise scale at normalized time t_norm.
        Using an exponential schedule: σ(t) = σ_min * (σ_max/σ_min)^t
        """
        if self.process_type == 'VE':
            return self.sigma_min * (self.sigma_max / self.sigma_min)**(t_norm)
        else:
            return None

    def score_fn(self, model, x, t):
        """
        Given the noise-predicting model, returns the score (i.e. ∇_x log p_t(x))
        at actual time t. Note that the model expects a normalized time (t/T).
        For VP: score = - (predicted noise) / sqrt(1 - ᾱ(t))
        For VE: score = - (predicted noise) / σ(t)
        """
        t_norm = t / self.T  # normalize to [0,1]
        if self.process_type == 'VP':
            alpha_bar = self.alpha_bar(t_norm).view(-1, *([1] * (x.dim() - 1)))
            
            epsilon = model(x, t_norm) #.view(-1, 1))
            score = -epsilon / torch.sqrt(1 - alpha_bar)
            return score
        elif self.process_type == 'VE':
            sigma_t = self.sigma_fn(t_norm).view(-1, *([1] * (x.dim() - 1)))
            epsilon = model(x, t_norm)#.view(-1, 1))
            score = -epsilon / sigma_t
            return score

    def forward(self, x_start, t_norm):
        """
        Forward (diffusion) process: given a clean sample x_start and time t (in [0,T]),
        returns the noised version x_t.
        For VP: x_t = sqrt(ᾱ(t)) x_start + sqrt(1-ᾱ(t)) noise
        For VE: x_t = x_start + σ(t)*noise
        """
        noise = torch.randn_like(x_start)
        if self.process_type == 'VP':
            alpha_bar = self.alpha_bar(t_norm).view(-1, *([1] * (x_start.dim() - 1)))
            x_t = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        elif self.process_type == 'VE':
            sigma_t = self.sigma_fn(t_norm).view(-1, *([1] * (x_start.dim() - 1)))
            x_t = x_start + sigma_t * noise
        return x_t, noise

    def training_losses(self, model, x_start, model_kwargs=None, **kwargs):
        """
        Training loss for the diffusion process.
        Samples t ~ Uniform(0, T), applies the forward process, and then
        computes the MSE loss between the network’s predicted noise and the true noise.
        """
        x_start = x_start.to(self.device)
        batch_size = x_start.size(0)
        # Sample t uniformly from [0, T]
        t = torch.rand(batch_size, device=self.device) * self.T
        t_norm = t / self.T
        x_t, noise = self.forward(x_start, t_norm)
        
        if model_kwargs is None:
            model_kwargs = {}
        # The model takes x_t and normalized time t_norm
        predicted_noise = model(x_t, t_norm, **model_kwargs)
        loss = F.mse_loss(predicted_noise, noise)
        return {'loss': loss}

    def sample(self, model, shape, get_sample_history=False, reverse_steps=1000, progress=True, deterministic = False, **kwargs):
        """
        SDE sampling using the Euler–Maruyama method to solve the reverse-time SDE:
          dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dẆ
        For VP:
          f(x,t) = -0.5 β(t)x  and  g(t) = sqrt(β(t))
          where β(t) is given either by a linear or cosine schedule.
        For VE:
          f(x,t) = 0  and  g(t) = σ(t)
        """
        # Initialize x_T (the prior sample)
        if self.process_type == 'VP':
            xt = torch.randn(shape, device=self.device)
        elif self.process_type == 'VE':
            t_norm = torch.tensor(1.0, device=self.device)
            sigma_T = self.sigma_fn(t_norm).view(*([1] * len(shape)))
            xt = sigma_T * torch.randn(shape, device=self.device)
        samples = []
        model.eval()
        with torch.inference_mode():
            # Create a time discretization from T to 0
            t_seq = torch.linspace(self.T, 0, reverse_steps + 1, device=self.device)
            if progress:
                progress_bar = tqdm(range(reverse_steps))
            else:
                progress_bar = range(reverse_steps)
            for i in progress_bar:
                t_current = t_seq[i]
                t_next = t_seq[i + 1]
                dt = t_next - t_current  # dt is negative (reverse time)
                # Create a batch of current time values for the update.
                t_batch = torch.full((shape[0],), t_current, device=self.device)
                t_norm_batch = t_batch / self.T

                if self.process_type == 'VP':
                    # Compute β(t) depending on schedule.
                    if self.schedule == 'linear':
                        beta_t = self.beta_min + t_norm_batch * (self.beta_max - self.beta_min)
                    elif self.schedule == 'cosine':
                        beta_t = (math.pi / (self.T * (1 + self.s))) * torch.tan(((t_norm_batch + self.s) / (1 + self.s)) * (math.pi / 2))
                    beta_t = beta_t.view(-1, *([1] * (xt.dim() - 1)))
                    f = -0.5 * beta_t * xt
                    g = torch.sqrt(beta_t)
                elif self.process_type == 'VE':
                    sigma_t = self.sigma_fn(t_norm_batch).view(-1, *([1] * (xt.dim() - 1)))
                    f = 0.0
                    g = sigma_t
                
                
                # Get the score (using the noise-predicting network)
                score = self.score_fn(model, xt, t_batch)
                # Euler–Maruyama update:
                #   x = x + [f - g^2 * score] dt + g * sqrt(-dt) * z,   where z ~ N(0, I)
                if not deterministic:
                    z = torch.randn_like(xt)
                    xt = xt + (f - (g**2) * score) * dt + g * torch.sqrt(-dt) * z
                else:
                    xt = xt + (f - (g**2) * score / 2) * dt
                if get_sample_history:
                    samples.append(xt.clone())
        return xt if not get_sample_history else torch.stack(samples)

#############################################
# functions for model/optimizer/device
#############################################

def create_mlp_model():
    
    model = SimpleModel.MLPModel(
        nfeatures = 2,
        time_emb_type = 'learnable',
        time_emb_size = 8,
        nblocks = 2,
        nunits = 32,
        skip_connection = True,
        layer_norm = True,
        dropout_rate = 0.1,
        learn_variance = False,
    )
    
    return model


def create_unet():
    channels = 1
    out_channels = 1
    
    first_layer_embedding = True
    embedding_dim = 3 # MD4 needs a value for masks, so set of values is {0, 1, 2}
    output_dim = 1 # We only output a single probability value
    
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=32,
            out_channels= out_channels,
            num_res_blocks=2,
            attention_resolutions= [2, 4],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= 0.0,
            channel_mult= [1, 2, 2, 2], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#
            num_heads=4,
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
            first_layer_embedding=first_layer_embedding,
            embedding_dim= embedding_dim,
            output_dim = output_dim,
        )
    return model

def create_model(dataset):
    """
    Returns a model for the specified dataset.
    """
    if dataset == "mnist":
        return create_unet()
    elif dataset == "gaussian_mixture":
        return create_mlp_model()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")



def create_optimizer(model, lr=1e-3):
    """
    Returns an Adam optimizer for the model.
    """
    return optim.AdamW(model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999))
    

def get_device():
    """
    Returns the available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


#############################################
# Data loader, with augmentation for MNIST
#############################################

def create_dataset(dataset_name):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == "gaussian_mixture":
        # Create a dataset with a Gaussian mixture distribution:
        # Two equal mixtures with means [-0.5, 0] and [0.5, 0], and standard deviation 0.1. 5000 samples.
        n_samples = 10000
        means = np.array([[-0.5, 0.0], [0.5, 0.0]])
        std = 0.1
        # Half the samples from each mixture component.
        samples1 = np.random.randn(n_samples // 2, 2) * std + means[0]
        samples2 = np.random.randn(n_samples - n_samples // 2, 2) * std + means[1]
        samples = np.concatenate([samples1, samples2], axis=0).astype(np.float32)
        # shuffle dataset
        np.random.shuffle(samples)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(samples), torch.zeros(len(samples)))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset
    

def create_dataloader(dataset, batch_size):
    """
    Returns a DataLoader for the specified dataset.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)




#############################################
# Training loop with checkpointing
#############################################

def train(num_epochs, checkpoint_interval, batch_size, learning_rate, checkpoint_dir, dataset):
    device = get_device()
    print("Using device:", device)

    # Create objects.
    diffusion_process = DiffusionProcess(device=device)
    model = create_model(dataset).to(device) # the model will depend on the dataset used
    # model = create_unet().to(device)
    optimizer = create_optimizer(model, lr=learning_rate)
    dataset = create_dataset(dataset)
    dataloader = create_dataloader(dataset, batch_size)

    model.train()
    epoch_losses = []
    for epoch in (range(1, num_epochs + 1)):
        running_loss = 0.0
        for batch_idx, (data, _) in (enumerate(dataloader)):
            
            data = data.to(device) 
            optimizer.zero_grad()

            # Compute the training loss.
            losses_dict = diffusion_process.training_losses(model, x_start=data)
            loss = losses_dict['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if batch_idx % 100 == 0:
            #     print(f"Epoch [{epoch}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

        # Save a checkpoint every checkpoint_interval epochs.
        if epoch % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
            }, checkpoint_path)
            print("Saved checkpoint to", checkpoint_path)

    print("Training finished.")

#############################################
# Generation procedure and image saving
#############################################

def generate(checkpoint_path, num_images, reverse_steps, get_samples_history, output_path, dataset_name):
    device = get_device()
    print("Using device:", device)

    # Create MD4Generation and the model.
    diffusion_process = DiffusionProcess(device=device)
    # the model will depend on the dataset used
    model = create_model(dataset_name).to(device)
    # model = create_unet().to(device)
    
    dataset = create_dataset(dataset_name)

    # Load the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.inference_mode():
        # For MNIST, shape=(num_images, 1, 32, 32); for gaussian_mixture, shape=(num_images, 2)
        shape = (num_images, 1, 32, 32) if dataset_name == "mnist" else (num_images, 2)
        print('Generating samples with shape:', shape)
        samples = diffusion_process.sample(
            model=model,
            shape=shape,
            reverse_steps=reverse_steps,
            get_sample_history=get_samples_history,
            deterministic=False,
            progress=True
        )

    # Save the generated images.
    save_images(samples, output_path, get_samples_history, dataset=dataset)
    print("Saved generated images to", output_path)

def save_images(samples, output_path, get_samples_history=False, dataset = None):
    """
    Save a grid of images to output_path. If the samples are 2D (as in a Gaussian mixture),
    we use matplotlib to create a scatter plot.
    If get_samples_history is True, also save the full history.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If samples are 2D points (either (N,2) or (steps, N,2)), use a scatter plot.
    float_samples = samples.clone().detach().float()
    
    if (float_samples.dim() == 2) or (float_samples.dim() == 3 and float_samples.shape[-1] == 2):
        # Assume shape (num_samples, 2) or (steps, num_samples, 2)
        if float_samples.dim() == 3:
            # Save final samples from the last time step.
            final_samples = float_samples[-1]
        else:
            final_samples = float_samples
        
        # clip finale samples to box [-lim, lim]\times [-lim, lim]
        box_lim = 1.2
        final_samples = torch.clamp(final_samples, -box_lim, box_lim)
        plt.figure(figsize=(6, 6))
        if dataset is not None:
            # retrieve some samples from the original dataset
            real_samples, _ = dataset[:final_samples.shape[0]]
            plt.scatter(real_samples[:, 0].cpu(), real_samples[:, 1].cpu(), s=10, alpha=0.4, label="Real samples")
        plt.scatter(final_samples[:, 0].cpu(), final_samples[:, 1].cpu(), s=10, alpha=0.4, label="Generated samples")
        plt.title("Generated 2D Gaussian Mixture")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-box_lim, box_lim)
        plt.ylim(-box_lim, box_lim)
        plt.legend()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        if get_samples_history and float_samples.dim() == 3:
            history_dir = os.path.splitext(output_path)[0] + "_history"
            os.makedirs(history_dir, exist_ok=True)
            for i, step_samples in enumerate(float_samples):
                plt.figure(figsize=(6, 6))
                plt.scatter(step_samples[:, 0].cpu(), step_samples[:, 1].cpu(), s=10, alpha=0.6)
                plt.title(f"Step {i}")
                plt.xlabel("x")
                plt.ylabel("y")
                step_path = os.path.join(history_dir, f"step_{i:04d}.png")
                plt.savefig(step_path)
                plt.close()
    else:
        # Otherwise, assume images and use torchvision’s utility.
        if get_samples_history:
            final_samples = float_samples[-1]
            vutils.save_image(final_samples, output_path, nrow=int(math.sqrt(final_samples.size(0))), normalize=True)
            history_dir = os.path.splitext(output_path)[0] + "_history"
            os.makedirs(history_dir, exist_ok=True)
            for i, step_samples in enumerate(float_samples):
                step_path = os.path.join(history_dir, f"step_{i:04d}.png")
                vutils.save_image(step_samples, step_path, nrow=int(math.sqrt(step_samples.size(0))), normalize=True)
        else:
            vutils.save_image(float_samples, output_path, nrow=int(math.sqrt(float_samples.size(0))), normalize=True)


#############################################
# Main entry point with argument parsing
#############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Diffusion Training and Generation Script")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands: train or generate")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "gaussian_mixture"])

    # Sub-parser for generation.
    gen_parser = subparsers.add_parser("generate", help="Generate images using a checkpoint")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate ")
    gen_parser.add_argument("--reverse_steps", type=int, default=100, help="Number of reverse diffusion steps (T)")
    gen_parser.add_argument("--get_samples_history", action="store_true", help="Save the full generation history")
    gen_parser.add_argument("--output_path", type=str, default="generated.png", help="Path to save the generated image grid")
    gen_parser.add_argument("--dataset", type=str, help="Dataset to use", choices=["mnist", "gaussian_mixture"])

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            dataset=args.dataset
        )
    elif args.command == "generate":
        generate(
            checkpoint_path=args.checkpoint_path,
            num_images=args.num_images,
            reverse_steps=args.reverse_steps,
            get_samples_history=args.get_samples_history,
            output_path=args.output_path,
            dataset_name=args.dataset
        )
    else:
        parser.print_help()