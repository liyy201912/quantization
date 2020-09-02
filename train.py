import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models.resnet_cifar import resnet18

# parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', action='store_true', default=True,
                    help='resume training')

args = parser.parse_args()
net = resnet18()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/liyanyu/cifar/data', train=True, download=False,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='/home/liyanyu/cifar/data', train=False, download=False,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = 10

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    net.to(device)
    if args.resume:
        net.load_state_dict(torch.load('checkpoints/resnet_4bit_quantized_cifar10_acc_92.35.pt'))
    torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    acc = 0

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        learning_rate = args.lr * (0.1 ** (epoch // 20))
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        running_loss = 0.0
        print('learning rate:', learning_rate)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # regularization for clipping a
            # decay = 1e-4
            # l2_reg = torch.tensor(0.0, requires_grad=True).cuda()
            # for name, param in net.named_parameters():
            #     if '.a' in name:
            #         l2_reg = l2_reg + torch.norm(param)

            loss = criterion(outputs, labels)
            # loss = loss + decay * l2_reg
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        # report test accuracy every epoch
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy = 100 * float(correct) / float(total)

        print('Accuracy on 10000 test images: %.2f %%' % accuracy)
        if accuracy > acc:
            acc = accuracy
            path = 'checkpoints/{arch}_{type}_quantized_cifar10_acc_{prec1:.2f}.pt' \
                .format(arch='resnet', type='4bit', prec1=acc)
            torch.save(net.state_dict(), path)
        print('Current best accuracy: %.2f %%' % acc)

    print('Finished Training')


if __name__ == "__main__":
    main()
