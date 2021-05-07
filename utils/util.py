import torch
import torch.nn as nn

class PatchEmbedd(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, inputs):
        x = self.proj(inputs) # output -> (batch, embed_dim, n_patchW, n_patchH)
        x = x.flatten(2)
        x = torch.transpose(x, 1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, n_heads=12):
        super().__init__()
        self.n_heads = n_heads

        


