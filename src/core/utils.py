
import wandb

class EpochModelCallBack:
    def __init__(self):
        pass

    def call(self, epoch:int, epoch_train_loss:float, epoch_train_acc:float, epoch_train_cer:float, epoch_val_loss:float, epoch_val_acc:float, epoch_val_cer:float):
        raise NotImplementedError()
    def __call__(self,epoch:int,  epoch_train_loss:float, epoch_train_acc:float, epoch_train_cer:float, epoch_val_loss:float, epoch_val_acc:float, epoch_val_cer:float):
        self.call(epoch, epoch_train_loss, epoch_train_acc, epoch_train_cer, epoch_val_loss, epoch_val_acc, epoch_train_cer)


class WandDBCallBack:
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


