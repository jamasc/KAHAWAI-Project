# this code creates a dataset with the features obtained by pretrained resnet50
# right now works for pukele sheet
# TODO make it with input args to define datasheet

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

from models import ResNet50Embedder
from dataset import ImagePathDataset

# --- Hyperparameters ---
excel_file = "../../data/pukele_datasheet.xlsx"  # CSV with 'path','label' columns
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_file = "../../features/pukele_embeddings.pt"

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load dataset and dataloader ---
dataset = ImagePathDataset(excel_file, transform=transform, new_root='/home/jans26/koa_scratch/streamflow/images')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Load model ---
model = ResNet50Embedder().to(device)
model.eval()

all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        embeddings = model(images)  # [batch, 2048]
        all_embeddings.append(embeddings.cpu())
        all_labels.extend(labels)

# Concatenate all embeddings
all_embeddings = torch.cat(all_embeddings, dim=0)
all_labels = torch.tensor(all_labels)

# Save embeddings and labels
torch.save({'embeddings': all_embeddings, 'labels': all_labels}, output_file)
print(f"Saved embeddings and labels to {output_file}")
