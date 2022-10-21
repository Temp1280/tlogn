# -*- coding: utf-8 -*-
import random
import time

import torch





a = time.time()

b = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))

print(a)
print(b)

import numpy as np

neighbor_ts = np.array([1, 2, 55, 77, 176, 177, 200])
tmp_time = 201
neighbor_ts = neighbor_ts - tmp_time  # 可以增加其他参数

def get_w(sample_ratio):
    weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
    weights = weights / sum(weights)
    print(weights)

def get_w2(sample_ratio):
    weights = 1.0 / np.abs(neighbor_ts)
    weights = np.power(weights, sample_ratio)
    weights = weights / sum(weights)
    print(weights)


get_w(0.5)
get_w(0.1)
get_w(0.2)
get_w(1.0)


get_w2(0.1)
get_w2(0.2)
get_w2(0.5)
get_w2(1.0)
get_w2(2.0)

a = torch.zeros((4,5))
b = torch.tensor([1,2,3,1])
a[torch.arange(4), b] = 1
print(a)

a = np.array([1,2,2,1,2,3])
print(np.nonzero(a==2)[0])
print(np.nonzero(a==1)[0])

a = np.nonzero(a==2)[0]
a = list(a)
random.shuffle(a)
print(a)
print(a[0:100])

for i in range(0,10,3):
    print(i)

a = [1,2,3,0]
print(sorted(a))


a = (1,2,3)
b = (2,3)
print(np.array([a,b]))

# a = [1,2,3]
# b = a[1,2,0]
# print(b)


items = list(range(0,10))
random.shuffle(items)
print(items)