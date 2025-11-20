import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def get_embeddings(embedder, images):
    with torch.no_grad():
        feats = embedder(images)  # [batch, 2048, 1, 1]
        feats = feats.view(feats.size(0), -1)  # flatten -> [batch, 2048]
    return feats

def save_model(model, path='water_classifier_head.pth'):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def show_image(img_tensor, title=""):
    """Helper function to display a single image tensor."""
    img = img_tensor.detach().cpu().permute(1, 2, 0)  # C,H,W -> H,W,C
    img = (img - img.min()) / (img.max() - img.min())  # normalize for display
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

def stratified_split(dataset, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    
    labels = [dataset[i][1] for i in range(len(dataset))]
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
    
        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset