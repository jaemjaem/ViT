import torch
from utils.util import PatchEmbedd
import torchvision
import torchvision.transforms as transforms

def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    patchEmbedd = PatchEmbedd(img_size=32, patch_size=16)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    for i, data in enumerate(trainloader):
        data, label = data
        ss = patchEmbedd(data)

if __name__ == '__main__':
    train()
