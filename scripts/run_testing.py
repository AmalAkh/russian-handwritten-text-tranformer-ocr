import sys
from pathlib import Path
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import Settings
from src.core.ocr_tranformer import OCRTransformer
from src.dataset.ocr_dataset import OCRDataset
from src.dataset.utils import create_char_mapping, get_max_lbl_len
from src.core.utils import untokenize_tensor_batch

def main():
    # --- Device Configuration ---
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"

    print(f"Using device: {device_name}")

    # --- Data Loading & Preprocessing ---
    labels_df = pd.read_csv("data/labels.csv", header=0)
    char_to_idx, idx_to_char = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "EOS", "SOS"])
    max_seq_len = get_max_lbl_len(labels_df["text"])

    dataset = OCRDataset("data/images", "data/filtered_labels.csv", char_to_idx, max_seq_len, resize=Settings.RESIZE_IMAGE)

    total_size = len(dataset)
    train_size = int(Settings.TRAIN_SPLIT * total_size)
    val_size = int(Settings.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Settings.RANDOM_STATE) 
    )

    # Create a dataloader with a batch size of exactly 32
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=Settings.NUM_WORKERS
    )

    # --- Model Initialization & Loading ---
    transformer = OCRTransformer(
        len(char_to_idx), 
        max_seq_len, 
        idx_to_char,
        d_model=Settings.D_MODEL, 
        pretrained_vit_model=Settings.PRETRAINED_VIT_ENCODER, 
        train_pretrained_vit_encoder=Settings.TRAIN_VIT_ENCODER, 
        device="cpu"
    )

    # Load model weights from file
    model_path = os.path.join("models", "model-0e.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        transformer.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Proceeding with initialized model.")

    # Convert model to mps (as required)
    transformer.device = device_name
    transformer.to(device_name)
    print(f"Model successfully converted and moved to: {device_name}")

    # --- Prediction on a Single 32-Batch & Output Comparison ---
    transformer.eval()
    with torch.no_grad():
        for x, y_text, y in test_dataloader:
            x = x.to(device_name)
            y = y.to(dtype=torch.long, device=device_name).contiguous()
            
            print(f"Processing a single batch with shape: {x.shape}\n")
            output = transformer.forward(x, torch.zeros((32, 1)).long().to("mps"))
            
            output_for_metrics = output.argmax(dim=2).int()
            
            # Untokenize predictions back to text
            decoded_predictions = untokenize_tensor_batch(
                output_for_metrics, 
                transformer.idx_to_char, 
                [transformer.eos_idx, transformer.pad_idx, transformer.sos_idx]
            )
            
            # Print side-by-side comparison of Ground Truth vs Predictions
            print(f"{'='*10} BATCH PREDICTIONS VS GROUND TRUTH {'='*10}")
            for idx, (pred_text, true_text) in enumerate(zip(decoded_predictions, y_text)):
                print(f"[{idx + 1:02d}] Ground Truth: '{true_text}'")
                print(f"      Prediction:   '{pred_text}'")
                print("-" * 50)
                
            break

    print("--- Testing completed successfully for one batch ---")

if __name__ == "__main__":
    main()