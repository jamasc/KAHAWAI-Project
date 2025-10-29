## Streamflow Dataset
# labels are converted as follows:
# 0 (label_1 sheet) to 2 (run present)
# 1 to 0 (dry bed)
# 2 to 1 (isolated pools)
# 3 to 3 (freshet)
# the last two labels ('poor quality' and 'not working') are discarded

from torch.utils.data import Dataset, random_split
from PIL import Image
import pandas as pd

binary_class_map = {0:2, 1:0, 2:1, 3:3}

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

