import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0

class VitEmbedding(nn.Module):
    def __init__(self, d_model:int=768, patch_size:int=16, img_size=(224,224)):
        super().__init__()

        self.num_patches = (img_size[0] // patch_size)*(img_size[1] // patch_size)
        
        model = models.efficientnet_b0(pretrained=True)

        self.conv = nn.Sequential(
            model.features[0:4]
        )

        self.class_token = nn.Parameter(torch.zeros((1,1,d_model)))
        self.pos_embedding = nn.Parameter(torch.randn((1, 28*28+1,d_model)))

        
    def forward(self, x):
        conv_output = self.conv(x)
        
        embedded = conv_output.flatten(2)
        embedded = embedded.transpose(1,2)
     
        class_tokens = self.class_token.expand(x.size()[0], -1, -1)

        embedded = torch.cat([embedded, class_tokens], dim=1)
        embedded += self.pos_embedding
      
        return embedded


if __name__ == "__main__":
    imgs = torch.ones((16,3,224,224))
    emb = VitEmbedding(d_model=40)
    emb(imgs)