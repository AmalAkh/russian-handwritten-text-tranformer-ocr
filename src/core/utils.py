
import wandb
import torch
from typing import List

class EpochModelCallBack:
    def __init__(self):
        pass

    def call(self, epoch:int, epoch_train_loss:float, epoch_train_acc:float, epoch_train_cer:float, epoch_val_loss:float, epoch_val_acc:float, epoch_val_cer:float):
        raise NotImplementedError()
    def __call__(self,epoch:int,  epoch_train_loss:float, epoch_train_acc:float, epoch_train_cer:float, epoch_val_loss:float, epoch_val_acc:float, epoch_val_cer:float):
        self.call(epoch, epoch_train_loss, epoch_train_acc, epoch_train_cer, epoch_val_loss, epoch_val_acc, epoch_train_cer)


class WandDBCallBack(EpochModelCallBack):
    def __init__(self,run_name:str, config:dict):
        self.run = wandb.init(
            name=run_name,
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="amal-akhmadinurov-stu",
            # Set the wandb project where this run will be logged.
            project="russian-cursive-vit",
            # Track hyperparameters and run metadata.
            config=config
        )
    def call(self, epoch:int, epoch_train_loss:float, epoch_train_acc:float, epoch_train_cer:float, epoch_val_loss:float, epoch_val_acc:float, epoch_val_cer:float):
        self.run.log({"epoch":epoch,
               "Train Loss":epoch_train_loss,
               "Train Accuracy":epoch_train_acc,
               "Train CER":epoch_train_cer,
               "Val Loss":epoch_val_loss,
               "Val Accuracy":epoch_val_acc,
               "Val CER":epoch_val_cer,
        })


def untokenize_tensor_batch(batch:torch.Tensor, idx_to_char:List[str], special_char:List[int]):
    decoded_batch = []
   
    
    # Ensure the batch is on CPU and converted to a standard list
    batch_list = batch.cpu().tolist()
   
    
    for seq in batch_list:
        # 1. Filter out special tokens
        # 2. Map integer to character
        # 3. Join tokens into a single string
        chars = [idx_to_char[idx] for idx in seq if idx not in special_char]
        decoded_batch.append("".join(chars))
        
    return decoded_batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)