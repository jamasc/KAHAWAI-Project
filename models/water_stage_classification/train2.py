import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import wandb
from tqdm import tqdm
import yaml
import argparse

from dataset import EmbeddingDataset
from models import ResNet50Embedder, ClassificationHead
from utils import get_embeddings, save_model, stratified_split


def main(config_path):
    # Load config from YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # --- Weights & Biases ---
    wandb.init(
        project = config['wandb_project'],     #"water_existence", 
        name = config['wandb_name'],         #"train_run 9/1", 
        config = {
            "learning_rate": config['learning_rate'],
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Dataset ---
    dataset = EmbeddingDataset(config['embedding_path']) 
    
    train_dataset, val_dataset, _ = stratified_split(dataset, config['val_split'], 1 - config['train_split'] - config['val_split'], seed=config['random_seed'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # --- Models ---
    classifier = ClassificationHead().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    
    # --- Training ---
    for epoch in range(config['epochs']):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for features, labels in train_bar:
            labels = labels - 1
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
        train_acc = correct / total
        train_loss = total_loss / total
    
        # Validation
        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                labels = labels - 1
                features, labels = features.to(device), labels.to(device)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
    
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
    
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc})
    
    # Save model
    save_model(classifier, config['model_save_path'])        #"classifier_head.pth"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)