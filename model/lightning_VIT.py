import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.util import Block, PatchEmbedd
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning.metrics.functional import accuracy

class LitVIT(pl.LightningModule):
    def __init__(self, data_dir='./data', learning_rate=1e-4, n_layers=12, img_size=32, patch_size=16, n_classes=10,
                 in_chans=3, embed_dim=768, n_heads=12, mlp_ratio=4.0):
        super().__init__()

        ###############
        # Train Block #
        ###############
        self.data_dir = data_dir
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        ###############
        # Model Block #
        ###############
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
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = self.patch_embedd(inputs)

        cls_token = self.cls_token.expand(batch, -1, -1)

        x = torch.cat((cls_token, x), dim=1)  # shape -> (batch, n_patch + 1, embed_dim)
        x = x + self.pos_embedding

        for block in self.encoder:
            x = block(x)

        x = self.layer_norm(x)
        final_cls_token = x[:, 0]
        outputs = self.head(final_cls_token)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        _, predicted = torch.max(outputs.data, 1)
        acc = accuracy(predicted, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=1024, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=10, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=10, num_workers=8)