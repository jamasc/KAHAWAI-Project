# this is a testing script to test saved models on a specific test dataset
# takes a model and a dataset as input
# if the dataset is not a designated test dataset it also does a split to create test dataset
# outputs are the amount of test cases (per label), the amount of correct predictions (per label), and the confidence (per label)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F

from dataset import ImagePathDataset
from models import ResNet50Embedder, ClassificationHead
from utils import get_embeddings, load_model, show_image

# Parameters
model_path = "pukele_classifier_head.pth"
data_sheet = "pukele_datasheet.xlsx"
is_designated_test_set = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Dataset ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImagePathDataset(data_sheet, transform=transform, new_root='/home/jans26/koa_scratch/streamflow/images')

if not is_designated_test_set:
    _, _, test_dataset = split_dataset(dataset, seed=42)
else:
    test_dataset = dataset
    
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Load Models ---
embedder = ResNet50Embedder().to(device)
classifier = load_model(ClassificationHead, model_path, device)
classifier.eval()

criterion = nn.CrossEntropyLoss()

# --- Evaluation ---
test_loss, test_correct, test_total = 0, 0, 0

label_counts = defaultdict(int)
label_correct = defaultdict(int)
label_confidences = defaultdict(list)

best_samples = {}
worst_samples = {}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = get_embeddings(embedder, images)
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)
        probs = F.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)
        
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            conf = confidences[i].item()
            img = images[i].cpu()

            label_counts[label] += 1
            if pred == label:
                label_correct[label] += 1
                label_confidences[label].append(conf)

            # Track best (highest confidence correct) and worst (lowest confidence incorrect)
            if pred == label:
                if label not in best_samples or conf > best_samples[label]["conf"]:
                    best_samples[label] = {"img": img, "pred": pred, "conf": conf}
            else:
                if label not in worst_samples or conf < worst_samples[label]["conf"]:
                    worst_samples[label] = {"img": img, "pred": pred, "conf": conf}

test_acc = test_correct / test_total
test_loss = test_loss / test_total

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print("\nPer-label results:")
for label in sorted(label_counts.keys()):
    total = label_counts[label]
    correct = label_correct[label]
    avg_conf = sum(label_confidences[label]) / len(label_confidences[label]) if label_confidences[label] else 0.0
    acc = correct / total if total > 0 else 0.0
    print(f"Label {label:>3}: "
          f"Total={total:>4}, "
          f"Correct={correct:>4}, "
          f"Accuracy={acc*100:6.2f}%, "
          f"Avg Confidence={avg_conf:.4f}")

# show best and worst prediction samples
for label in sorted(label_counts.keys()):
    plt.figure(figsize=(6, 3))
    plt.suptitle(f"Label {label}")

    if label in best_samples:
        plt.subplot(1, 2, 1)
        s = best_samples[label]
        show_image(s["img"], title=f"Best (Pred={s['pred']}, Conf={s['conf']:.2f})")

    if label in worst_samples:
        plt.subplot(1, 2, 2)
        s = worst_samples[label]
        show_image(s["img"], title=f"Worst (Pred={s['pred']}, Conf={s['conf']:.2f})")

    plt.tight_layout()
    plt.show()
