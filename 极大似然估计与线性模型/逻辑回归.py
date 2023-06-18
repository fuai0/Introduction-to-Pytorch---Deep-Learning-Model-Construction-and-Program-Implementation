from sklearn.datasets import load_iris
import torch
from torch import nn as nn
from torch import optim
import matplotlib.pyplot as plt
#
# iris = load_iris()
#
# x = iris.data[:100]
# y = iris.target[:100]
# x = torch.tensor(x,dtype = torch.float32)
# y = torch.tensor(y,dtype = torch.float32)
#
# net = nn.Linear(4,1)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(),lr=0.15)
#
# losses = []
# for epoch in range(100):
#     optimizer.zero_grad()
#
#     y_pred = net(x)
#
#     loss = loss_fn(y_pred.view_as(y),y)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
#
# plt.plot(losses)
# plt.show()




#
from sklearn.datasets import load_digits
digits = load_digits()

x = digits.data
y = digits.target

x = torch.tensor(x,dtype = torch.float32)
y = torch.tensor(y,dtype = torch.int64)

net = nn.Linear(x.size()[1],10)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(),lr=0.01)

losses = []
for epoch in range(100):
    optimizer.zero_grad()

    y_pred = net(x)

    loss = loss_fn(y_pred, y)
    loss.backward()

    optimizer.step()
    losses.append(loss.item())

_, y_pred = torch.max(net(x), 1)
print(y_pred)