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

def untokenize(tokens:List[int], idx_to_char:List[str]):
    
    return "".join([idx_to_char[token] for token in tokens])

def remove_speical_chars(tokens:List[int], special_char:List[int]):   
    return list(filter(lambda ch: ch not in special_char, tokens))

def pad_sequence(seq:List[int],max_len:int, pad_token:int=0):

    return seq+([pad_token]*(max_len-len(seq)))

def get_max_lbl_len(labels:List[str]):
    return max([len(lbl) for lbl in labels])

def filter_non_existant_files(labels_df:pd.DataFrame, data_dir:str|Path):
    df = labels_df.copy()
    df["exists"] = labels_df["file_name"].apply( lambda x: os.path.exists(os.path.join(data_dir, str(x)))) == True
    return df[df["exists"]]


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent  


    labels_df = pd.read_csv("data/labels.csv", header=0)
    print(len(labels_df))

    print(pad_sequence([1,2,3], 10))

    filtered_df = filter_non_existant_files(labels_df, "data/images")
    filtered_df.to_csv("data/filtered_labels.csv")
    print(len(filtered_df))
    char_to_idx, idx_to_char = create_char_mapping(labels_df["text"].tolist(), ["<PAD>", "<EOS>", "<SOS>"])
    print(len(idx_to_char))
    print(char_to_idx)
    print(tokenize("Привет, мир", char_to_idx))

