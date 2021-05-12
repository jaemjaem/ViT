import torch
import torch.nn as nn
from utils.util import Block, PatchEmbedd

class VIT(nn.Module):
    def __init__(self, n_layers=12, img_size=32, patch_size=16, n_classes=10,
                 in_chans=3, embed_dim=768, n_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embedd = PatchEmbedd(img_size=img_size, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim)

        # need cls token + position embedding(cls + n_patch -> 1 + n_patch)
        self.cls_token = nn.parameter.Parameter(torch.zeros((1, 1, embed_dim)))
        self.pos_embedding = nn.parameter.Parameter(torch.zeros((1, 1 + self.patch_embedd.n_patches, embed_dim)))

        # N of Blocks -> Base Model N = 12, n_heads = 12, hidden_size = 768, MLP size = 3072
        self.encoder = nn.ModuleList(
            [
                Block(embed_dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = self.patch_embedd(inputs)

        cls_token = self.cls_token.expand(batch, -1, -1)

        x = torch.cat((cls_token, x), dim=1) # shape -> (batch, n_patch + 1, embed_dim)
        x = x + self.pos_embedding

        for block in self.encoder:
            x = block(x)

        final_cls_token = x[:, 0]
        return self.head(final_cls_token)