## Streamflow Dataset
# labels are converted as follows:
# 0 (label_1 sheet) to 2 (run present)
# 1 to 0 (dry bed)
# 2 to 1 (isolated pools)
# 3 to 3 (freshet)
# the last two labels ('poor quality' and 'not working') are discarded

import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import pandas as pd
import os

binary_class_map = {0:2, 1:0, 2:1, 3:3}

# dataset of labeled images from DAR TODO: remake
class BinaryStreamFlowDataset(Dataset):
    def __init__(self, excel_file, transform=None, limit=float('inf')):
        self.data = []
        xls = pd.ExcelFile(excel_file)

        # each sheet = one label
        for idx, sheet_name in enumerate(xls.sheet_names):
            if idx > 3: break
            label = binary_class_map[idx]
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            for img_path in df['Image_Path'].dropna():
                img_path = img_path.replace('D:', '/home/jans26/koa_scratch/streamflow/images')
                img_path = img_path.replace('\\', '/')
                self.data.append((img_path, label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# dataset to get images and labels from datasheet
class ImagePathDataset(Dataset):
    """
    Dataset that reads image paths and labels from an Excel sheet.
    Optionally replaces old drive roots (e.g., D:/, E:/) with a new root directory.
    """
    def __init__(
        self,
        excel_file,
        img_col="path",
        label_col="top_5_percent",
        transform=None,
        new_root=None,
        old_roots=("D:/", "E:/")
    ):
        """
        Args:
            excel_file (str): Path to Excel file containing image paths and labels.
            img_col (str): Name of the column containing image paths.
            label_col (str): Name of the column containing labels.
            transform (callable, optional): Optional torchvision transform.
            new_root (str, optional): Replace old root (e.g., 'D:/') with this new root.
            old_roots (tuple, optional): Roots to replace, e.g. ('D:/', 'E:/').
        """
        self.data = pd.read_excel(excel_file)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.new_root = new_root
        self.old_roots = old_roots

    def __len__(self):
        return len(self.data)

    def _replace_root(self, path):
        """Replace old drive roots with the new root if specified."""
        if self.new_root is None:
            return path

        # Normalize slashes for consistency
        path = path.replace("\\", "/")
        for old_root in self.old_roots:
            old_root = old_root.replace("\\", "/")
            if path.startswith(old_root):
                rel_path = path[len(old_root):].lstrip("/")
                return os.path.normpath(os.path.join(self.new_root, rel_path))
        # If no match, return path unchanged
        return path

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, self.img_col]
        label = self.data.loc[idx, self.label_col]

        img_path = self._replace_root(img_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# dataset that works with saved embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_file):
        data = torch.load(embeddings_file)
        self.embeddings = data['embeddings']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
