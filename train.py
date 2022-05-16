# -*- coding: utf-8 -*-
# Time : 2022/5/12 11:15 PM
# Author : sk-w
# Email : 15734082105@163.com
# File : train.py
# Project : Lenet-Pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import LeNet

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# loadData
traindata = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=transform
)

# loadData
valdata = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# dataloader
trainloader = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valdata, batch_size=4, shuffle=True, num_workers=4)

# test
testValIter = iter(valloader)
test_img, test_label = testValIter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img/2 + 0.5
#     img = img.numpy()
#     plt.imshow(np.transpose(img,(1,2,0)))
#     plt.show()
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_img))

net = LeNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# train
for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimizer
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print
        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_img)
                # [batch,10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train loss: %.3f test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Traing')
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
