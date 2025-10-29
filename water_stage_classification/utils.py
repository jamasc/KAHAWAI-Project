import torch

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