import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics

import numpy as np
import matplotlib.pyplot as plt
import time
import random

from net import MyNet, Loss, Self_Attension_Net, Net
from block import  ConvBlock
from dataload import dataloader_with_uni_channels
from config import Config


config = Config()
# batch_size * in_channels * time_length
# x = torch.randn(4, 3, 10)

# net = MyNet()

# y = net(x)

# x = torch.randn(4, 4, )
# print(y)

# net = nn.RNN(3, 7, 2)
# x = torch.randn(5,10,3)
# state = torch.randn()
# y,_ = net(x)
# print(y.shape,_.shape)

# class MultiChannelGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_channels):
#         super(MultiChannelGRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_channels = num_channels
        
#         # 创建单个 GRU 层
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
#     def forward(self, input_data):
#         # input_data 的维度应该是 (batch_size, num_channels, sequence_length, input_size)
#         batch_size = input_data.size(0)
#         sequence_length = input_data.size(2)
#         print(f"==>> input_data.shape: {input_data.shape}")
        
        

#         # 将通道维度和序列维度展平
#         input_data = input_data.view(batch_size * self.num_channels, sequence_length, self.input_size)
#         print(f"==>> input_data.shape: {input_data.shape}")
        
        
#         # 初始化隐藏状态
#         # h0 = torch.zeros(self.num_layers, batch_size * self.num_channels, self.hidden_size).to(input_data.device)
        
#         # GRU 前向计算
#         output, _ = self.gru(input_data)
#         print(f"==>> _.shape: {_.shape}")
#         temp = _[:,:,:]
        
#         # 重新调整输出形状
#         output = _.view(batch_size, self.num_channels, self.hidden_size)
#         print(f"==>> output.shape: {output.shape}")

#         print((torch.all(torch.eq(output[0,:,:],temp[:,0:3,:]))))
        
        
#         return output

# # 示例用法
# input_size = 1
# hidden_size = 20
# num_layers = 1
# num_channels = 3
# sequence_length = 5
# batch_size = 32

# # 创建输入数据
# input_data = torch.randn(batch_size, num_channels, sequence_length, input_size)

# # 创建模型实例
# model = MultiChannelGRU(input_size, hidden_size, num_layers, num_channels)

# # 前向计算
# output = model(input_data)
# print("Output shape:", output.shape)


# net = ConvBlock(1,32)
# x = torch.randn(2,5, 12)
# y = net(x)
# print(f"==>> y.shape: {y.shape}")

# loss = Loss()
# y_p = torch.randn(24)
# lable = (torch.randn(6), torch.randn(8))

# y = loss(y_p, lable)
# print(f"==>> y: {y}")

# net = MyNet()
# x1 = torch.randn(7,5, 100)
# x2 = torch.randn(7,5,100)
# y = net(x1, x2)
# print(f"==>> y: {y}")
# print(f"==>> y.shape: {y.shape}")



def process_each_channel(data, nums_task = 20,nums_sup = 20,nums_que = 20,time_length = 100):
    # data 单变量时间序列
    sup_data = []
    que_data = []
    sup_label = []
    que_label = []

    mx_time = data.shape[0]
    
    for _ in range(nums_task):
        sup_label_tmp = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4
        que_label_tmp = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4
        sup_data_tmp = []
        que_data_tmp = []

        time_start_list = random.sample(range(mx_time-time_length+1), nums_sup + nums_que)
        data_sample = [data[i:i + time_length] for i in time_start_list]
        sup_data_tmp = data_sample[0:nums_sup]
        que_data_tmp = data_sample[nums_sup:nums_que + nums_sup]


        for label in sup_label_tmp:
            if label == 0:
                pass
            elif label == 1:
                pass
            elif label == 2:
                pass
            else:
                pass 
        
        for label in que_label_tmp:
            if label == 0:
                pass
            elif label == 1:
                pass
            elif label == 2:
                pass
            else:
                pass 

        sup_data.append(sup_data_tmp)
        que_data.append(que_data_tmp)
        sup_label.append(sup_label_tmp)
        que_label.append(que_label_tmp)

    return sup_data, que_data, sup_label, que_label

# a = torch.tensor(torch.randn(2,2,3))
# print(a)
# print(a.shape)
# layer = nn.LayerNorm(3)
# b = layer(a)
# print(b)

# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         attn_weights = torch.matmul(q, k.transpose(1, 2))
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)
#         attended_values = torch.matmul(attn_weights, v)
#         return attended_values


# net = SelfAttention(32)
# x = torch.randn(10,7,32)
# y = net(x)
# print(y.shape)

# a = torch.tensor([0,1,2,2])
# b = torch.tensor([0,1,2,3])
# c = torch.tensor([0,2,2,5])
# m = [1]
# d = sklearn.metrics.classification_report(a, b)
# f = sklearn.metrics.classification_report(b, c)
# print(d)
# print("---------")
# print(f)


# y_p = np.random.randint(0,10000,(400,))
# label_sup = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4


# que_predict = np.zeros((20,4))
# for i, dist in enumerate(y_p):
#     x = int(i / 20)
#     y = i % 20

#     que_predict[y][label_sup[x]] += dist
# que_predict[:, 0] /= 8
# que_predict[:, 1:] /= 4
# print(que_predict)
# que_pred = que_predict.argmax(1)
# print(que_pred)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_contrastive = Loss()

val_loss, val_acc = 0, 0
classification_report_val_real = []
classification_report_val_pred = []

sup_test = []
que_test = []
sup_label_test = []
que_label_test = []
# path_val = ['../raw_data/4.npy']

# for p in path_val:
#     st, qt, sl, ql = dataloader_with_uni_channels(p,nums_task = config.nums_task_val, time_length=config.time_length)
#     st = np.array(st)
#     qt = np.array(qt)
#     sl = np.array(sl)
#     ql = np.array(ql)

#     st = torch.tensor(st,dtype=torch.float)
#     qt = torch.tensor(qt,dtype=torch.float)
#     sl = torch.tensor(sl)
#     ql = torch.tensor(ql)

# #     sup_train.append(st)
#     sup_test += st
#     que_test += qt
#     sup_label_test += sl
#     que_label_test += ql

sup_test = np.load("../data/sup_test.npy", allow_pickle=True)
que_test = np.load("../data/que_test.npy", allow_pickle=True)
sup_label_test = np.load("../data/sup_label_test.npy", allow_pickle=True)
que_label_test = np.load("../data/que_label_test.npy", allow_pickle=True)


net = Self_Attension_Net(time_length=config.time_length)
net.load_state_dict(torch.load("../model/best.pth"))
net.to(device)

net.eval()
net.double()
n = len(sup_test)
for i in range(n):
    sup_input = sup_test[i][:,:,:].astype(float)
    sup_input = torch.tensor(sup_input).to(device)
    que_input = que_test[i][:,:,:].astype(float)
    que_input = torch.tensor(que_input).to(device)
    sup_l = sup_label_test[i][:]
    sup_l = torch.tensor(sup_l).to(device)
    que_l = que_label_test[i][:]
    que_l = torch.tensor(que_l).to(device)

    outputs = net(sup_input, que_input)
    loss, pred, label_que, que_pred = loss_contrastive(outputs, (sup_l, que_l), False)
    # val_loss += loss.item()
    # val_acc += pred.item()

    label_que.cpu().numpy()
    # que_pred.cpu().numpy()         
    classification_report_val_real = np.hstack((classification_report_val_real, label_que))
    classification_report_val_pred = np.hstack((classification_report_val_pred, que_pred))

            


# print(f"val_loss:{val_loss/n}")
# print(f"val_acc:{val_acc/n}")
print(sklearn.metrics.classification_report(classification_report_val_real, classification_report_val_pred))