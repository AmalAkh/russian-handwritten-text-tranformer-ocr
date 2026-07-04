from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd 
from torchvision import transforms


from torchvision.io import decode_image
import torch
import os
from PIL import Image
from utils import tokenize


class OCRDataset(Dataset):

    def __init__(self, images_dir_path:str|Path, labels_path:str|Path, char_to_idx:dict):
        self.images_dir_path = images_dir_path
        self.labels_path = labels_path
        self.char_to_idx = char_to_idx

        self.tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((50, 150))
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

        if image is None:
            image = Image.open(os.path.join(self.images_dir_path, self.image_filenames[index])).convert("RGB")
            image = self.tranform(image)
            self.images[index] = image
        
        if label is None:
            text = self.labels_df["text"][index]
            label = tokenize(text, self.char_to_idx)
            label = torch.Tensor(label)
            self.labels[index] = label
            

        return image, label
    
            
        
        


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent  

    from utils import create_char_mapping
    labels_df = pd.read_csv("data/labels.csv", header=0)
    char_to_idx, _ = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "EOS", "SOS"])

    dataset = OCRDataset("data/images", "data/labels.csv", char_to_idx)

    # Test retrieving the first item
    img, lbl = dataset[0]
    print("Image shape:", img.shape)
    print("Label tensor:", lbl)



    

        