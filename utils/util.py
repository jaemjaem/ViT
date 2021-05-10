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
        x = x.flatten(2) # shape : (batch, embed_dim, n_patch)
        x = torch.transpose(x, 1, 2) # shape : (batch, n_patch, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = dim ** 0.5

        self.make_Q = nn.Linear(dim, dim) # make Q
        self.make_K = nn.Linear(dim, dim) # make K
        self.make_V = nn.Linear(dim, dim) # make V

        self.proj = nn.Linear(dim, dim) # linear projection after Scaled Dot-Product Attention outputs are concated

    def split_head(self, QKV, input_shape):
        batch, n_patch, _ = input_shape
        QKV = QKV.reshape(batch, n_patch, self.n_heads, self.head_dim)
        output = QKV.permute(0, 2, 1, 3)
        return output

    def forward(self, inputs):
        input_shape = inputs.shape

        # Make Query, Key, Value
        query = self.make_Q(inputs) # output shape -> (batch, img_patch_size + class_patch_size, dim)
        key = self.make_K(inputs)
        value = self.make_V(inputs)

        # Split heads for Multi-Head Attention
        query = self.split_head(query, input_shape) # output shape -> (batch, n_heads, n_patch, head_dim)
        key = self.split_head(key, input_shape)
        value = self.split_head(value, input_shape)

        # Scaled Dot-Product Attention
        key_T = key.transpose(3, 2)
        scale_mat = torch.matmul(query, key_T) * (1 / self.scale) # output shape -> (batch, n_head, n_patch, n_patch)
        attn_mat = scale_mat.softmax(dim=-1) # Score between patches

        sdp_attn_output = torch.matmul(attn_mat, value) # output shape -> (batch, n_head, n_patch, head_dim)
        sdp_attn_output = sdp_attn_output.transpose(1, 2) # output shape -> (batch, n_patch, n_head, head_dim)
        sdp_attn_output = sdp_attn_output.flatten(2) # flatten (n_head, head_dim) for MLP, output shape -> (batch, n_patch, dim)

        # Wo Linear
        output = self.proj(sdp_attn_output)

        return output


class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6) # epsilon = 1e-6 why ?
        self.attention = Attention(embed_dim, n_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_size = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_size)

    def forward(self, inputs):
        x = self.attention(self.layer_norm1(inputs)) + inputs
        output = self.mlp(self.layer_norm2(x)) + x
        # x = self.attention(inputs) + inputs
        # output = self.mlp(x) + x
        return output