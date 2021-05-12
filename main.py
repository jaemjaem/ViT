import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import pytorch_lightning as pl

from model.VIT import VIT
from model.lightning_VIT import LitVIT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1024
save_path = './checkpoints'

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=8,pin_memory=True)


    validset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=10,
                                             shuffle=False, num_workers=2)

    net = VIT(n_layers=1, patch_size=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001)

    for epoch in range(1000):  # 데이터셋을 수차례 반복합니다.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            net.train()
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data[0].to(device), data[1].to(device)
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()

            if i % 25 == 24:  # print every 2000 mini-batches
                print('[%d, %5d/%d] loss: %.3f' %
                      (epoch + 1, i + 1, len(trainloader), running_loss / 25))
                running_loss = 0.0
        if epoch % 50 == 49:
            torch.save(net.state_dict(), './checkpoints/%d.pth' % (epoch + 1))
        # validation part
        correct = 0
        total = 0
        for i, data in enumerate(valid_loader, 0):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
              (epoch + 1, 100 * correct / total)
              )

    print('Finished Training')

def test():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    validset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=10,
                                               shuffle=False, num_workers=2)

    net = VIT(n_layers=1, patch_size=4).to(device)
    net.load_state_dict(torch.load('./checkpoints/200.pth'))
    net.eval()

    for epoch in range(1000):  # 데이터셋을 수차례 반복합니다.
        correct = 0
        total = 0
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
              (epoch + 1, 100 * correct / total)
              )

if __name__ == '__main__':
    train()
    # ####### pytorch - lightning
    # model = LitVIT(data_dir='./data', n_layers=1, patch_size=16)
    # trainer = pl.Trainer(gpus=1, max_epochs=20, progress_bar_refresh_rate=20)
    # trainer.fit(model)

