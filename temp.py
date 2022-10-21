# -*- coding: utf-8 -*-
import numpy as np
import torch


a = np.array([1,2,3,4,5])
# b = np.array([4,2])
# c = np.argwhere(a == 3)
#
# index_ = [item in b for item in a]
# print(np.arange(len(a))[index_])
# print(c)

b = [1,2]
a[b] = 10
print(a)

a = torch.randn(4, 4)
print(a)
b = torch.argsort(a, dim=1, descending=True)
c = torch.argsort(b, dim=1, descending=False)
print(c)

# a = torch.randn((256,8000,200))
# print(a.shape[0])

a = torch.tensor([[1,2,3],[4,2,6]])
b = torch.tensor([[1, 1],[6, 4]])
c = a

a = a.unsqueeze(-1)  # BM1
b = b.unsqueeze(1)  # B1N
index = a == b
print(index)
index = torch.sum(index, dim=-1)
print(index)

c[index != 0] = -1
print(c)
a, b = c.size()
print(a)
print(b)

a = torch.randn(4,3)
print(a)
a = a.expand(2,4,3)
print(a)

a = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.01, 0.0],
                            [0.0, 0.0, 1.2, 0.0],
                            [0.0, 0.0, 0.0,-0.4]])
b, c = torch.nonzero(a, as_tuple=True)
print(a[b, c])

batch = 4
batch_data_array = np.array([1,88,5,6])
batch_list = np.array(range(batch), dtype=np.long)
h_list = batch_data_array
r_list = np.ones(batch, dtype=np.long) * -1
t_list = batch_data_array
ts_list = np.zeros(batch, dtype=np.long)

temp_array = np.stack([batch_list, h_list, r_list, t_list, ts_list], axis=1)
print(temp_array)

a = torch.tensor([1,2,3,-1,1,-1])
print(a == -1)
print(torch.nonzero(a == -1).squeeze(1))
print('--'*100)

# node_list = np.arange(10)
# print(node_list)
#
#
# from torch_scatter import scatter
#
# src = torch.randn(6, 4)
# index = torch.tensor([0, 1, 0, 1, 2, 1])
# print(src)
# # Broadcasting in the first and last dim.
# out = scatter(src, index, dim=0, reduce="sum")
# print(out)
# print(out.size())

# a = np.random.randn(5,2)
# print(a)
# print([(item[0], item[1]) for item in a])
#
#
# a = torch.randn(4, 4)
# b = torch.max(a, 1, keepdim=True)[0]
# print(a)
# print(b)
#
# c = a[[torch.arange(4), torch.arange(4)]]
# print(c)
# print(b+c+b)
#
# device = torch.device('cuda:0')
# a = torch.randn(100,100).to(device)
# print(a)

# a = list([2, 13, 0])
# a.remove(0)
# print(a)
#
# a = np.array([[2,13,0],[1, 0]])
# b = a[np.array([0,0,1,1,0])]
# print(b)
# c = [list(item).remove(0) for item in b]
#
#
# print(b)
# print(c)
#
# a = torch.LongTensor([0,0,1,1])
# print(a == 0)
# print(torch.nonzero(a==0))
#
# a = torch.arange(10)
# print(torch.topk(a, 10))
import torch.nn as nn

param = nn.Parameter(torch.Tensor(4,1))
nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('tanh'))
print(param)

a = torch.tensor([1,2,3])
print(a)
print(a.squeeze())


a = torch.randn(3,5)

a = torch.relu(a)
print(a)
b = torch.max(a, dim=-1, keepdim=True).values
print(b)
print(a/b)

a = torch.randn(3,5)
print(a)

b_range = torch.arange(a.shape[0])
obj = torch.LongTensor([2,3,1])
b = a[b_range, obj]
print(b)
c = torch.tensor([[1,0,0,0,0], [0,1,0,0,0],[0,0,1,0,0]])
print(a[c==0].reshape(3,-1))
print(c==0)