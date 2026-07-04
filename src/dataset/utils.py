import os
from typing import List, Tuple, Dict, Union
from pathlib import Path
import pandas as pd
def create_char_mapping(labels:List[str], special_chars:List[str]=[])->Tuple[Dict[str, int],list]:
    
    uniq_chars = set()
    for label in labels:
        for ch in label:
            uniq_chars.add(ch)
    
    idx_to_char = [*special_chars]
    char_to_idx = {idx:ch for idx, ch in enumerate(special_chars)}

    prefix = len(idx_to_char)
    for idx, char in enumerate(uniq_chars):
        idx_to_char.append(char)
        char_to_idx[char] = prefix+idx

    return char_to_idx, idx_to_char            

def tokenize(word: str, char_to_idx: Union[Dict[str, int], List[str]]) -> List[int]:
    if isinstance(char_to_idx, list):
        char_to_idx = {char: idx for idx, char in enumerate(char_to_idx)}
    
    if not isinstance(word, str):
        return []
        
    return [char_to_idx[char] for char in word if char in char_to_idx]


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent  


    labels_df = pd.read_csv("data/labels.csv", header=0)
    char_to_idx, idx_to_char = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "<EOS>", "<SOS>"])
    print(len(idx_to_char))
    print(char_to_idx)
    print(tokenize("Привет, мир", char_to_idx))