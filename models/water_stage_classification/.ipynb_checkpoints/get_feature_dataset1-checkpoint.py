# this code creates a dataset with the features obtained by pretrained resnet50
# works for DAR and pukele (see v0)
# call in interactive jupyter lab session on KOA from terminal

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

from models import ResNet50Embedder
from dataset import ImagePathDataset

# --- Hyperparameters ---
excel_files = [
    "../../../data/DAR_1000_stratified.xlsx",
    "../../../data/DAR_1001_stratified.xlsx",
    "../../../data/DAR_1002_stratified.xlsx",
    "../../../data/DAR_1003_stratified.xlsx",
    "../../../data/DAR_1004_stratified.xlsx",
    "../../../data/DAR_1005_stratified.xlsx",
    "../../../data/DAR_1006_stratified.xlsx",
]
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_files = [
    "../../../features/DAR_1000_stratified_embeddings.pt",
    "../../../features/DAR_1001_stratified_embeddings.pt",
    "../../../features/DAR_1002_stratified_embeddings.pt",
    "../../../features/DAR_1003_stratified_embeddings.pt",
    "../../../features/DAR_1004_stratified_embeddings.pt",
    "../../../features/DAR_1005_stratified_embeddings.pt",
    "../../../features/DAR_1006_stratified_embeddings.pt",
]

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load model ---
model = ResNet50Embedder().to(device)
model.eval()

for i, excel_file in enumerate(excel_files):
    # --- Load dataset and dataloader ---
    dataset = ImagePathDataset(excel_file, transform=transform, new_root='/home/jans26/koa_scratch/streamflow/images')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Processing file {i+1}/{len(excel_files)}"):
            images = images.to(device)
            embeddings = model(images)  # [batch, 2048]
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.tensor(all_labels)

    # Save embeddings and labels
    torch.save({'embeddings': all_embeddings, 'labels': all_labels}, output_files[i])
    print(f"Saved embeddings and labels to {output_files[i]}")
