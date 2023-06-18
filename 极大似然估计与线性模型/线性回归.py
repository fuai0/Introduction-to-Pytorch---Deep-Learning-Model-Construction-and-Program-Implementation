#

# import torch
#
# w_true = torch.Tensor([1, 2, 3])
#
# x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
#
# y = torch.mv(x, w_true) + torch.randn(100) * 0.5
#
# w = torch.randn(3, requires_grad=True)
#
# gamma = 0.1
#
# losses = []
#
# for epoc in range(100):
#     w.grad = None
#
#     y_pred = torch.mv(x, w)
#
#     loss = torch.mean((y - y_pred)**2)
#     loss.backward()
#     w.data = w.data - gamma * w.grad.data
#     print(w.grad)
#     losses.append(loss.item())
#
# from matplotlib import pyplot as plt
#
# plt.plot(losses)
# plt.show()
# import torch
#


#
# import torch
# from torch import optim
# import torch.nn as nn
# import matplotlib.pyplot as plt
#
# w_true = torch.Tensor([1, 2, 3])
# x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
# y = torch.mv(x, w_true) + torch.randn(100) * 0.5
#
# net = nn.Linear(in_features=3,out_features=1,bias=True)
#
# opimizer = optim.SGD(net.parameters(),lr=0.1)
# loss_fn = nn.MSELoss()
# losses = []
# for epoch in range(100):
#     opimizer.zero_grad()
#
#     y_pred = net(x)
#
#     loss = loss_fn(y_pred.view_as(y),y)
#     loss.backward()
#     opimizer.step()
#     losses.append(loss.item())
#
# plt.plot(losses)
# plt.show()
# print(list(net.parameters()))




#



