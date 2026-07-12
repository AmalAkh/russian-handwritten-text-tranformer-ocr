from transformers import ViTModel, ViTConfig
import torch 
import torch.nn as nn
from torch.nn import TransformerDecoderLayer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CharErrorRate
from src.core.utils import EpochModelCallBack

from tqdm import tqdm

class OCRTransformer(nn.Module):

    def __init__(self, vocab_size:int,max_seq_len:int, pad_idx:int=0, sos_idx:int=1, eos_idx:int=2, 
                d_model:int=1024, dim_feedforward=2048, num_decoder_heads:int=16, num_decoder_layers:int=8,
                loss_fn=nn.CrossEntropyLoss(), pretrained_vit_model='google/vit-large-patch32-384', train_pretrained_vit_encoder=True, device:str="cpu"):
        super().__init__()

        self.loss_fn = loss_fn
        self.device = device

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.encoder = ViTModel.from_pretrained(pretrained_vit_model)

        if not(train_pretrained_vit_encoder):
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        decoder_layer = TransformerDecoderLayer(d_model,num_decoder_heads, dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.output_layer = nn.Linear(d_model, vocab_size)

        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters())


        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size).to(device=self.device)
        self.cer = CharErrorRate().to(device=self.device)



    def forward(self, img:torch.Tensor, target_label:torch.Tensor):
        
        encoded_input = self.encoder(img).last_hidden_state

        #print(encoded_input.size())
        
        if target_label.size()[1] == 1:
            print("do")
            finished = torch.zeros(target_label.size(0), dtype=torch.bool, device=target_label.device)
            for i in range(0, self.max_seq_len):

                embedded_target = self.embedding(target_label)+self.pos_embedding[:, :target_label.size(1), :]

                target_mask = nn.Transformer.generate_square_subsequent_mask(target_label.size(1)).to(target_label.device)
                target_padding_mask = (target_label == self.pad_idx)

                decoder_output = self.decoder(
                    tgt=embedded_target,
                    memory= encoded_input,
                    tgt_mask= target_mask,
                    tgt_key_padding_mask= target_padding_mask
                )

                logits = self.output_layer(decoder_output)
                
                next_token = logits[:,-1,:].argmax(1).unsqueeze(1)
                finished |= (next_token.view(-1) == self.eos_idx)


                target_label = torch.cat([target_label, next_token], dim=1)
                if finished.all():
                    return target_label
                else:
                    return torch.cat([target_label, torch.tensor([[self.eos_idx]]).to(target_label.device)], dim=1)


        else:

            embedded_target = self.embedding(target_label)+self.pos_embedding[:, :target_label.size(1), :]

            target_mask = nn.Transformer.generate_square_subsequent_mask(target_label.size(1)).to(target_label.device)
            target_padding_mask = (target_label == self.pad_idx)

            decoder_output = self.decoder(
                tgt=embedded_target,
                memory= encoded_input,
                tgt_mask= target_mask,
                tgt_key_padding_mask= target_padding_mask
            )

            return self.output_layer(decoder_output)
    
    def evaluate(self, dataloader:DataLoader):
        
        with torch.no_grad():
            
            self.accuracy.reset()
            self.cer.reset()
            val_loss = 0

            for idx, (x,y) in enumerate(dataloader):
                
                x = x.to(self.device)

                y =  y.to(dtype=torch.long, device=self.device).contiguous()
                output = self.forward(x, y)
               

                output_for_metrics = output.argmax(dim=2).int()
             
                # compute metrics
                self.accuracy(output_for_metrics, y)
                self.cer(output_for_metrics, y)

                loss = self.loss_fn(output.permute(0, 2, 1), y)

                val_loss += loss.item()

               


        val_loss /= len(dataloader)
        epoch_accuracy = self.accuracy.compute().item()
        epoch_cer = self.cer.compute().item()

        return val_loss, epoch_accuracy, epoch_cer
            

    def fit(self, train_dataloader:DataLoader, val_dataloader:DataLoader, epochs:int=10, epoch_callback:EpochModelCallBack=None):

        for epoch in range(0, epochs):

            epoch_loss = 0

            self.accuracy.reset()
            self.cer.reset()

            for idx, (x, y) in enumerate(tqdm(train_dataloader)):
               
                x,y = x.to(self.device), y.to(self.device)
                y = y.long()
                output = self.forward(x, y)
               

                output_for_metrics = output.argmax(dim=2).int()
             
                # compute metrics
                self.accuracy(output_for_metrics, y)
                self.cer(output_for_metrics, y)

                loss = self.loss_fn(output.permute(0, 2, 1), y)

                epoch_loss += loss.item()

                
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss /= len(train_dataloader)
            epoch_accuracy = self.accuracy.compute().item()
            epoch_cer = self.cer.compute().item()

            val_loss, val_accuracy, val_cer = self.evaluate(val_dataloader)

            if epoch_callback is not None:
                epoch_callback(epoch, epoch_loss, epoch_accuracy, epoch_cer, val_loss, val_accuracy, val_cer)

            print(f"Epoch #{epoch+1}: Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_accuracy:.4f} Train CER: {epoch_cer:.4f} Val Loss: {val_loss} Val Accuracy: {val_accuracy} Val CER: {val_cer}")



import torch
import torch.nn as nn

# Assuming OCRTranformer is imported here

def run_smoke_test():
    print("--- Starting OCRTransformer Smoke Test ---")
    
    # 1. Define dummy hyperparameters
    BATCH_SIZE = 2
    VOCAB_SIZE = 50
    MAX_SEQ_LEN = 30
    
    # Note: 'google/vit-large-patch32-384' explicitly expects 384x384 inputs
    IMAGE_C, IMAGE_H, IMAGE_W = 3, 384, 384 
    TARGET_SEQ_LEN = 15

    print(f"Configuration: Batch={BATCH_SIZE}, Vocab={VOCAB_SIZE}, Max Seq={MAX_SEQ_LEN}")

    # 2. Initialize Model
    try:
        model = OCRTransformer(
            vocab_size=VOCAB_SIZE, 
            max_seq_len=MAX_SEQ_LEN,
            d_model=1024 # Must match ViT-large hidden size
        )
        # Put in eval mode to disable dropout (if any) for deterministic testing
        # model.eval() # Requires inheriting from nn.Module!
        print("Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 3. Create Dummy Data
    # Images: (N, C, H, W)
    dummy_images = torch.randn(BATCH_SIZE, IMAGE_C, IMAGE_H, IMAGE_W)
    
    # Labels: (N, S). Must be torch.long for nn.Embedding
    dummy_labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TARGET_SEQ_LEN), dtype=torch.long)
    
    # 4. Run Forward Pass
    try:
        print("Running forward pass...")
        output = model.forward(dummy_images, dummy_labels)
        
        # 5. Validate Output Shape
        expected_shape = (BATCH_SIZE, TARGET_SEQ_LEN, VOCAB_SIZE)
        assert output.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output.shape}"
        
        print(f"✅ Smoke test passed! Output shape is correct: {output.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed with error:\n{type(e).__name__}: {e}")

if __name__ == "__main__":
    run_smoke_test()