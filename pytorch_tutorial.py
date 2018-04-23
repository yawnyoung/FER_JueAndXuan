from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.autograd as autograd
import matplotlib.pyplot as plt
from PIL import Image


def imshow(img):
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cnn_example():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(Variable(images).cuda())

    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTM(1, 64)
        self.lstm2 = nn.LSTM(64, 1)

    def forward(self, seq, hc = None):

        if hc == None:
            hc1, hc2 = None, None
        else:
            hc1, hc2 = hc

        out, hc1 = self.lstm1(seq, hc1)
        out, hc2 = self.lstm2(out, hc2)

        return out,  (hc1, hc2)


if __name__ == '__main__':
    lstm = Sequence()
    lstm.cuda()
    # criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    # for i in range(1000):
    #     data = np.sin(np.linspace(0,10,100)+2*np.pi*np.random.rand())
    #     xs = data[:-1]
    #     ys = data[1:]
    #     x = Variable(torch.FloatTensor(xs).view(-1, 1, 1).cuda())
    #     y = Variable(torch.FloatTensor(ys).cuda())
    #
    #     optimizer.zero_grad()
    #     lstm_out, _ = lstm(x)
    #     loss = criterion(lstm_out[20:].view(-1), y[20:])
    #     loss.backward()
    #     optimizer.step()
    #
    #     if i % 10 == 0:
    #         print("i {}, loss {}".format(i, loss.data.numpy()[0]))





