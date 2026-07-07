import sys
from pathlib import Path

# 1. Get the directory of the current script (scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Go one level up to get the project root
PROJECT_ROOT = SCRIPT_DIR.parent

# 3. Add the project root to Python's path so it can find the 'src' folder
sys.path.append(str(PROJECT_ROOT))


from src.core.ocr_tranformer import OCRTransformer
from src.dataset.ocr_dataset import OCRDataset
from src.dataset.utils import create_char_mapping, get_max_lbl_len
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

print(f"Using device: {device_name}")

labels_df = pd.read_csv("data/labels.csv", header=0)
char_to_idx, _ = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "EOS", "SOS"])
max_seq_len = get_max_lbl_len(labels_df["text"])


# Initialize your full dataset
dataset = OCRDataset("data/images", "data/filtered_labels.csv", char_to_idx, max_seq_len)

# 1. Define split sizes (e.g., 80% train, 10% validation, 10% test)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Captures any remaining remainder

# 2. Split the dataset randomly
# Adding a generator with a manual seed ensures your splits are reproducible 
train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42) 
)

# 3. Create DataLoaders for each split
# It's standard practice to shuffle the training data, but not val/test data
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False)

print(f"Total dataset size:  {total_size}")
print(f"Training set size:   {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")
print("-" * 30)


train_dataloader = DataLoader(dataset, batch_size=12)

transformer = OCRTransformer(len(char_to_idx), max_seq_len, device=device_name)

transformer.fit(train_dataloader, val_dataloader)
print(transformer.evaluate(val_dataloader))
