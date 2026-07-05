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
from torch.utils.data import DataLoader

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


dataset = OCRDataset("data/images", "data/filtered_labels.csv", char_to_idx, max_seq_len)



train_dataloader = DataLoader(dataset, batch_size=12)

transformer = OCRTransformer(len(char_to_idx), max_seq_len)

transformer.fit(train_dataloader, train_dataloader)
