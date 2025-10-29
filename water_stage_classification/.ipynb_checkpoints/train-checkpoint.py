import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import wandb
from tqdm import tqdm

from dataset import BinaryStreamFlowDataset
from models import ResNet50Embedder, ClassificationHead
from utils import get_embeddings, save_model


def main():
    # --- Weights & Biases ---
    wandb.init(
        project="water_existence", 
        name="train_run 9/1", 
        config={
            "learning_rate": 1e-3,
            "epochs": 10,
            "batch_size": 32
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = BinaryStreamFlowDataset("/home/jans26/koa_scratch/streamflow/data/image_inventory_cam_1000.xlsx", transform=transform)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
    
    # --- Models ---
    embedder = ResNet50Embedder().to(device)
    classifier = ClassificationHead().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=wandb.config.learning_rate)
    
    # --- Training ---
    for epoch in range(wandb.config.epochs):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            embeddings = get_embeddings(embedder, images)
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
        train_acc = correct / total
        train_loss = total_loss / total
    
        # Validation
        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = get_embeddings(embedder, images)
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
    
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
    
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc})
    
    # Save model
    save_model(classifier, "classifier_head.pth")

if __name__ == "__main__":
    main()