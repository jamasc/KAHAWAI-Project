# models.py
import torch.nn as nn
import torchvision.models as models

class ResNet50Embedder(nn.Module):
    """
    Pretrained ResNet-50 backbone that outputs 2048-dim embeddings.
    """
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # remove final classification layer

    def forward(self, x):
        return self.model(x)  # [batch, 2048]


class ClassificationHead(nn.Module):
    """
    A small feedforward classifier on top of 2048-dim embeddings.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
