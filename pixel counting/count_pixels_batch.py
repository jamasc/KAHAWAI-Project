# this code takes one input image path and 1+ input strings, calls CLIPSeg, and returns the amount of pixels per input string as dict

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch

# CLIP Model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model inference
def countPixels(texts, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    pixel_counts = (probs > 0.5).int().sum(dim=(1, 2)).tolist() 

    out_dict = {}
    for i, count in enumerate(pixel_counts):
        out_dict[texts[i]] = count

    return out_dict