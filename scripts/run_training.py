import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from src.config.settings import Settings
from src.core.ocr_tranformer import OCRTransformer
from src.dataset.ocr_dataset import OCRDataset
from src.dataset.utils import create_char_mapping, get_max_lbl_len
from src.core.utils import WandDBCallBack, count_parameters

# --- Device Configuration ---
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

wandb_callback = WandDBCallBack("test_run_1", Settings.to_dict())


print(f"Using device: {device_name}")

# --- Data Loading & Preprocessing ---
labels_df = pd.read_csv("data/labels.csv", header=0)
char_to_idx, idx_to_char = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "EOS", "SOS"])

max_seq_len = get_max_lbl_len(labels_df["text"])

# Initialize your full dataset
dataset = OCRDataset("data/images", "data/filtered_labels.csv", char_to_idx, max_seq_len, resize=Settings.RESIZE_IMAGE)

# 1. Define split sizes using Settings
total_size = len(dataset)
train_size = int(Settings.TRAIN_SPLIT * total_size)
val_size = int(Settings.VAL_SPLIT * total_size)
test_size = total_size - train_size - val_size  # Captures any remaining remainder

# 2. Split the dataset randomly using Settings for reproducibility
train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(Settings.RANDOM_STATE) 
)

# 3. Create DataLoaders for each split using Settings
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=Settings.BATCH_SIZE, 
    shuffle=True, 
    num_workers=Settings.NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=Settings.BATCH_SIZE, 
    shuffle=False, 
    num_workers=Settings.NUM_WORKERS
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=Settings.BATCH_SIZE, 
    shuffle=False, 
    num_workers=Settings.NUM_WORKERS
)

print(f"Total dataset size:  {total_size}")
print(f"Training set size:   {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")
print("-" * 30)

# --- Model Initialization & Training ---
transformer = OCRTransformer(len(char_to_idx), max_seq_len, idx_to_char,
                            d_model=Settings.D_MODEL,
                            num_decoder_heads=14,
                            device=device_name)
print(f"Parameters: {count_parameters(transformer)}")
print(device_name)
transformer.fit(train_dataloader, val_dataloader, epochs=Settings.EPOCHS, epoch_callback=wandb_callback)
