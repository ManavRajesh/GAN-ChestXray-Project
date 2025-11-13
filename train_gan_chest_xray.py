#!/usr/bin/env python3
"""
train_gan_chest_xray.py

A self-contained PyTorch DCGAN-style training script for generating chest X-ray images.

Usage:
    python train_gan_chest_xray.py --data_root "/path/to/chest-xray-pneumonia" --epochs 50

Notes:
- Expects data folder with structure:
    <data_root>/train/NORMAL, <data_root>/train/PNEUMONIA
  (ImageFolder-compatible)
- Output images and checkpoints are saved under ./outputs/
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils

# --------------------------
# Utilities & Config
# --------------------------
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_output_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

# --------------------------
# Model definitions (DCGAN-like for 1-channel images)
# --------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super().__init__()
        # input is Z, going into a convolution
        self.net = nn.Sequential(
            # input: Z (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf//2, nc, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh()
        )

    def forward(self, z):
        # z: (B, nz)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (nc) x 128 x 128
            nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 1).squeeze(1)  # (B,)

# --------------------------
# Weight init
# --------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --------------------------
# Training Loop
# --------------------------
def train(args):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # transforms: grayscale, resize, to tensor, normalize to [-1,1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # dataset and dataloader (ImageFolder expects subfolders for classes)
    train_dir = os.path.join(args.data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}. Please point --data_root to the dataset root containing 'train/' subfolder.")

    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(f"Loaded dataset with {len(dataset)} images, classes: {dataset.classes}")

    nz = args.nz
    netG = Generator(nz=nz, ngf=args.ngf, nc=1).to(device)
    netD = Discriminator(nc=1, ndf=args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(args.sample_size, nz, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # outputs
    make_output_dirs(args.out_dir)
    start_epoch = 0

    # optionally resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        netG.load_state_dict(ckpt['netG'])
        netD.load_state_dict(ckpt['netD'])
        optimizerG.load_state_dict(ckpt['optG'])
        optimizerD.load_state_dict(ckpt['optD'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from checkpoint {args.resume}, starting epoch {start_epoch}")

    print("Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        for i, (imgs, _) in enumerate(dataloader):
            netD.zero_grad()
            imgs = imgs.to(device)
            b_size = imgs.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # train with real
            output = netD(imgs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(real_label)  # generator wants discriminator to think fakes are real
            output2 = netD(fake)
            errG = criterion(output2, label)
            errG.backward()
            D_G_z2 = output2.mean().item()
            optimizerG.step()

            if i % args.log_interval == 0:
                print(f"[{epoch}/{args.epochs}] Batch {i}/{len(dataloader)} \
                      Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Save sample images
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        sample_path = os.path.join(args.out_dir, "samples", f"epoch_{epoch:04d}.png")
        # Denormalize from [-1,1] to [0,1] for saving
        utils.save_image((fake_samples + 1) / 2.0, sample_path, nrow=8)
        print(f"Saved sample images to {sample_path}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optimizerG.state_dict(),
            'optD': optimizerD.state_dict(),
        }
        ckpt_path = os.path.join(args.out_dir, "checkpoints", f"ckpt_epoch_{epoch:04d}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # final save
    final_path = os.path.join(args.out_dir, "checkpoints", "final.pth")
    torch.save({
        'netG': netG.state_dict(),
        'netD': netD.state_dict()
    }, final_path)
    print("Training complete. Final models saved to", final_path)


# --------------------------
# CLI and main
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN on Chest X-Ray dataset (grayscale).")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root (contains train/ folder).")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save outputs & checkpoints.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam.")
    parser.add_argument("--nz", type=int, default=100, help="Dimensionality of the latent noise vector.")
    parser.add_argument("--ngf", type=int, default=128, help="Generator feature map base size.")
    parser.add_argument("--ndf", type=int, default=128, help="Discriminator feature map base size.")
    parser.add_argument("--sample_size", type=int, default=64, help="Number of sample images to produce for each epoch.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker threads.")
    parser.add_argument("--log_interval", type=int, default=50, help="Batches between logging.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    # Expand data_root if you passed ~
    args.data_root = os.path.expanduser(args.data_root)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir = os.path.join(os.path.expanduser(args.out_dir), f"run-{timestamp}")
    print("Output directory:", args.out_dir)
    train(args)
