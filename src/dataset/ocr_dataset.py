from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd 
from torchvision import transforms
import sys
from pathlib import Path
from typing import Tuple

from torchvision.io import decode_image
import torch
import os
from PIL import Image
from src.dataset.utils import tokenize, pad_sequence


class OCRDataset(Dataset):

    def __init__(self, images_dir_path:str|Path, labels_path:str|Path, char_to_idx:dict,max_seq_len:int, resize:Tuple[int,int]=(384,384)):
        self.images_dir_path = images_dir_path
        self.labels_path = labels_path
        self.char_to_idx = char_to_idx

        self.max_seq_len = max_seq_len

        self.tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize)
        ])
        

        self.labels_df = pd.read_csv(labels_path, header=0)
        
        self.image_filenames = self.labels_df["file_name"]
      

        self.labels = [None]*len(self.image_filenames)
        self.images = [None]*len(self.image_filenames)
        
        print(self.labels_df)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        text = self.labels_df["text"][index]

        if image is None:
            image = Image.open(os.path.join(self.images_dir_path, self.image_filenames[index])).convert("RGB")
            image = self.tranform(image)
            self.images[index] = image
        
        if label is None:
            label = tokenize(text, self.char_to_idx)
            label = pad_sequence(label, self.max_seq_len)
            label = torch.tensor(label, dtype=torch.int32)
            self.labels[index] = label
            

        return image, text, label
    
            
        
        


if __name__ == "__main__":


    from utils import create_char_mapping
    labels_df = pd.read_csv("data/labels.csv", header=0)
    char_to_idx, _ = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "EOS", "SOS"])

    dataset = OCRDataset("data/images", "data/labels.csv", char_to_idx)

    # Test retrieving the first item
    img, lbl = dataset[0]
    print("Image shape:", img.shape)
    print("Label tensor:", lbl)



    

        