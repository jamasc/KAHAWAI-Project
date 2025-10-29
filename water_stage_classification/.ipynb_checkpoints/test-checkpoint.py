import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import BinaryStreamFlowDataset
from models import ResNet50Embedder, ClassificationHead
from utils import get_embeddings, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Dataset ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = BinaryStreamFlowDataset("image_inventory_cam_1000.xlsx", transform=transform)
_, _, test_dataset = split_dataset(dataset, seed=42)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Load Models ---
embedder = ResNet50Embedder().to(device)
classifier = load_model(ClassificationHead, "classifier_head.pth", device)
classifier.eval()

criterion = nn.CrossEntropyLoss()

# --- Evaluation ---
test_loss, test_correct, test_total = 0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = get_embeddings(embedder, images)
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
test_loss = test_loss / test_total
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
