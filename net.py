import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from block import ConvBlock, GRUBlock

class MyNet(nn.Module):
    """Some Information about MyNet"""
    def __init__(self,time_length=100):
        super(MyNet, self).__init__()
        self.time_length = time_length

        self.net = nn.Sequential(
#             GRUBlock(input_size = 1, hidden_size = 8),
            ConvBlock(in_channels=1, out_channels=32,mid_channels=32,time_length=self.time_length),
            nn.Flatten(),
            nn.Linear(self.time_length*32,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,32),
#             nn.ReLU(inplace=True),
#             nn.Linear(256,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,4)
        )

        self.act = nn.Sigmoid()
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)
        
    def forward(self, sup_inp, que_inp):
#         def lcm(a, b):
#             return abs(a * b) // math.gcd(a, b)
        # sup_inp que_inp  [batch_size, in_channels, time_length]
        bz_sup = sup_inp.shape[0]
        bz_que = que_inp.shape[0]

        y_sup = self.net(sup_inp) #  [bz_sup,5]
        y_que = self.net(que_inp) #  [bz_que,5]

#         mx_len = lcm(bz_sup, bz_que)
        mx_len = bz_sup * bz_que

        y_sup = y_sup.repeat(int(mx_len / bz_sup),1)
        y_que = y_que.repeat(int(mx_len / bz_que),1)
        
        y_dist = F.pairwise_distance(y_sup,y_que,p=2)
        mean = torch.mean(y_dist)
        std = torch.std(y_dist)
        y_dist = (y_dist - mean) / (std + 1e-10)
#         print(y_dist)
        y_p = self.act(y_dist)

        return y_p


class Loss(torch.nn.Module):
    """
    loss function.
    """

    def __init__(self, margin=0.1, weight = 0.7, eps = 1e-10):
        super(Loss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.eps = eps

    def forward(self, y_p, label):
#         print(torch.mean(y_p))
        # y_p [lcm(bz_sup, bz_que)] from 0 ~ 1 靠近1表示越不相似
        # label ([bz_sup], [bz_que]) from [0, 1] 1 表示异常 0表示正常

        label_sup = label[0]
        label_que = label[1]
        mx_len = y_p.shape[0]
        # print(f"==>> mx_len: {mx_len}")
        # print(f"==>> label_que.shape[0]: {label_que.shape[0]}")
        # print(f"==>> label_sup.shape[0]: {label_sup.shape[0]}")
        # print(label_sup.repeat(int(mx_len / label_sup.shape[0])).shape)

        con_label = label_sup.repeat(int(mx_len / label_sup.shape[0])) - label_que.repeat(int(mx_len / label_que.shape[0]))

        con_label = torch.abs(con_label) # 1 表示不相似， 0 表示相似
#         print(con_label)
#         print(y_p)
#         print(con_label)
        # print(f"==>> con_label.shape: {con_label.shape}")
        # print(f"==>> y_p.shape: {y_p.shape}")
#         loss =  (1 - con_label) * y_p + con_label * (1 - y_p) * self.weight
        loss = - (1 - con_label) * torch.log(1 - y_p + self.eps) - con_label * torch.log(y_p + self.eps)
#         loss = F.pairwise_distance(con_label, y_p, p=2)
#         loss = (1-con_label) * torch.pow(y_p, 2) + (con_label) * torch.pow(torch.clamp(self.margin - y_p, min=0.0), 2)

        loss = torch.mean(loss)
        
        pred = torch.where(y_p > 0.5,0.0,1.0)
        pred = torch.mean(torch.abs(pred - con_label))
        
#         for y, pred in zip(con_label, y_p):
#             print(y.item(), pred.item())
        return loss, pred
    
