# simclr_pretrain.py

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os
import argparse

from lightly.data import LightlyDataset
from lightly.models import SimCLR
from lightly.transforms import SimCLRTransform
from lightly.loss import NTXentLoss
from tqdm import tqdm

# === Custom crop transform ===
def crop_bottom(img: Image.Image, pixels: int = 150):
    return img.crop((0, 0, img.width, img.height - pixels))

class BottomCropTransform:
    def __init__(self, pixels=150):
        self.pixels = pixels

    def __call__(self, img: Image.Image):
        return crop_bottom(img, self.pixels)

# === Collate function for SimCLR ===
def get_custom_collate_fn(crop_pixels=150, input_size=224):
    simclr_view = SimCLRTransform(input_size=input_size)
    crop_transform = BottomCropTransform(pixels=crop_pixels)

    def collate_fn(batch):
        x0, x1 = [], []
        for img, *_ in batch:
            cropped = crop_transform(img)
            view1, view2 = simclr_view(cropped)
            x0.append(view1)
            x1.append(view2)
        return torch.stack(x0), torch.stack(x1)
    
    return collate_fn

# === Training script ===
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & Dataloader
    dataset = LightlyDataset(input_dir=args.data_dir)
    collate_fn = get_custom_collate_fn(crop_pixels=args.crop_pixels, input_size=args.input_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Model
    backbone = resnet18(pretrained=False)
    backbone.fc = torch.nn.Identity()  # Remove classification head
    model = SimCLR(backbone, num_ftrs=512)
    model.to(device)

    # Loss and Optimizer
    loss_fn = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x0, x1 in pbar:
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0), model(x1)
            loss = loss_fn(z0, z1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {total_loss/len(dataloader):.4f}")

    # Save the model
    torch.save(model.backbone.state_dict(), args.checkpoint_path)
    print(f"Saved pretrained encoder to {args.checkpoint_path}")

# === CLI entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised SimCLR pretraining")
    parser.add_argument("--data_dir", type=str, default="images", help="Path to image directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_streamflow_ai.pth")
    parser.add_argument("--crop_pixels", type=int, default=150, help="Pixels to crop from bottom")
    parser.add_argument("--input_size", type=int, default=224, help="Input size for model")

    args = parser.parse_args()
    main(args)
