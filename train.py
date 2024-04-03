import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import random

from net import MyNet, Loss
from block import GRUBlock, ConvBlock
from dataload import dataloader
from config import Config





config = Config()

path = ['../pro_data/3_4.npy','../pro_data/4_4.npy']
sup_train = []
que_train = []
sup_label = []
que_label = []

for p in path:
    st, qt, sl, ql = dataloader(p,nums_task = config.nums_task_train, time_length=config.time_length)
    st = torch.tensor(st,dtype=torch.float)
    qt = torch.tensor(qt,dtype=torch.float)
    sl = torch.tensor(sl)
    ql = torch.tensor(ql)

#     sup_train.append(st)
    sup_train += st
    que_train += qt
    sup_label += sl
    que_label += ql



#     if sup_train is None:
#         sup_train, que_train, sup_label, que_label = st, qt, sl, ql
#     else:
#         sup_train = torch.cat([sup_train, st], dim=0)
#         que_train = torch.cat([que_train, qt], dim=0)
#         sup_label = torch.cat([sup_label, sl], dim=0)
#         que_label = torch.cat([que_label, ql], dim=0)
       
sup_test, que_test, sup_label_test, que_label_test = dataloader('../pro_data/4_4.npy',nums_task = config.nums_task_val,time_length=config.time_length)
sup_test = torch.tensor(sup_test,dtype=torch.float)
que_test = torch.tensor(que_test,dtype=torch.float)
sup_label_test = torch.tensor(sup_label_test)
que_label_test = torch.tensor(que_label_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MyNet(time_length=config.time_length).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=config.lr)
loss_contrastive = Loss()
# animator = d2l.Animator(xlabel='epoch', ylabel='loss',
#                             legend=['train'], xlim=[10, 500])

# loop over the dataset multiple times
train_list, val_list, train_acc_list, val_acc_list = [], [],[],[]
for epoch in range(config.nums_epoch):
    
# #     打乱训练数据
#     random.shuffle(sup_train)
#     random.shuffle(que_train)
#     random.shuffle(sup_label)
#     random.shuffle(que_label)
    
    
    running_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0
    
    net.train()
    for i in range(config.nums_task_train * len(path)):
        sup_input = sup_train[i][:,:,:].to(device)
        que_input = que_train[i][:,:,:].to(device)
        sup_l = sup_label[i][:].to(device)
        que_l = que_label[i][:].to(device)

        optimizer.zero_grad()

        outputs = net(sup_input, que_input)
        loss, pred = loss_contrastive(outputs, (sup_l, que_l))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_acc += pred.item()
            
    print(f"{epoch} train_loss:{running_loss/config.nums_task_train/len(path)}")
    train_list.append(running_loss/config.nums_task_train/len(path))
    print(f" train_acc:{train_acc/config.nums_task_train/len(path)}")
    train_acc_list.append(train_acc/config.nums_task_train/len(path))
    
    net.eval()
    with torch.no_grad():
        for i in range(config.nums_task_val):
#             sup_input = sup_train[i,:,:,:].to(device)
#             que_input = que_train[i,:,:,:].to(device)
#             sup_l = sup_label[i,:].to(device)
#             que_l = que_label[i,:].to(device)
            sup_input = sup_test[i,:,:,:].to(device)
            que_input = que_test[i,:,:,:].to(device)
            sup_l = sup_label_test[i,:].to(device)
            que_l = que_label_test[i,:].to(device)

            outputs = net(sup_input, que_input)
            loss, pred = loss_contrastive(outputs, (sup_l, que_l))
            val_loss += loss.item()
            val_acc += pred.item()
            
        print(f"val_loss:{val_loss/config.nums_task_val}")
        val_list.append(val_loss/config.nums_task_val)
        print(f"val_acc:{val_acc/config.nums_task_val}")
        val_acc_list.append(val_acc/config.nums_task_val)

    
    
def draw_loss(train_loss, val_loss, train_acc, val_acc, stride=10, path = './'):
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    
    epoch = range(0, len(train_loss), stride)
    
    tr_loss = train_loss[epoch]
    plt.plot(epoch,tr_loss,'k-',label='train_loss')
    
    val_loss = val_loss[epoch]
    plt.plot(epoch,val_loss,'r-',label='val_loss')
    
    train_acc = train_acc[epoch]
    plt.plot(epoch,train_acc,'k:',label='train_acc')
    
    val_acc = val_acc[epoch]
    plt.plot(epoch,val_acc,'r:',label='val_acc')
    
    plt.legend()
    
    plt.savefig(f"{path}{int(time.time())}.png")
    
draw_loss(train_list, val_list,train_acc_list, val_acc_list, stride = 10,path='../results/')

train_loss = np.array(train_list)
val_loss = np.array(val_list)
train_acc = np.array(train_acc_list)
val_acc = np.array(val_acc_list)

np.save(file="../results/train_loss.npy", arr=train_loss)
np.save(file="../results/val_loss.npy", arr=val_loss)
np.save(file="../results/train_acc.npy", arr=train_acc)
np.save(file="../results/val_acc.npy", arr=val_acc)