# this is a testing script to test saved models on a dataset, it uses the same datasplit as train2
# takes a model and a dataset as input
# outputs are the amount of test cases (per label), the amount of correct predictions (per label), and the confidence (per label)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from dataset import EmbeddingDataset
from models import ResNet50Embedder, ClassificationHead
from utils import get_embeddings, load_model, stratified_split

def main(config_path):
    # Load config from YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Dataset ---
    dataset = EmbeddingDataset(config['embedding_path']) 

    _, _, test_dataset = stratified_split(dataset, config['val_split'], 1 - config['train_split'] - config['val_split'], seed=config['random_seed'])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- Load Classifier ---
    classifier = load_model(ClassificationHead, config['model_save_path'], device)
    classifier.eval()

    criterion = nn.CrossEntropyLoss()

    # --- Evaluation ---
    test_loss, test_correct, test_total = 0, 0, 0

    label_counts = defaultdict(int)
    label_correct = defaultdict(int)
    label_confidences = defaultdict(list)

    best_samples = {}
    worst_samples = {}
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, labels in test_loader:
            labels = labels - 1
            # compute prediction and confidences
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
            y_true.extend(labels)
            y_pred.extend(predicted)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                conf = confidences[i].item()

                label_counts[label] += 1
                if pred == label:
                    label_correct[label] += 1
                    label_confidences[label].append(conf)

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
        
    results = pd.DataFrame({
        "label": y_true,
        "predicted": y_pred
    })

    return results    #, test_acc, test_loss, label_counts, label_correct, label_confidences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    main(args.config)