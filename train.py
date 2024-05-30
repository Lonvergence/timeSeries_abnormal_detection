import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics

import numpy as np
import matplotlib.pyplot as plt
import time
import random

from net import MyNet, Loss, Net, Self_Attension_Net
from block import  ConvBlock
from dataload import dataloader_with_uni_channels
from config import Config

config = Config()
NEED_RENEW_DATA = False
sup_train = []
que_train = []
sup_label = []
que_label = []

sup_test = []
que_test = []
sup_label_test = []
que_label_test = []

if NEED_RENEW_DATA:



    path_train = ['../raw_data/1.npy','../raw_data/2.npy','../raw_data/3.npy','../raw_data/4.npy']
    # path_train = ['../raw_data/1.npy']


    for p in path_train:
        st, qt, sl, ql = dataloader_with_uni_channels(p,nums_task = config.nums_task_train, time_length=config.time_length)
        # st = np.array(st, dtype=float)
        # qt = np.array(qt, dtype=float)
        # sl = np.array(sl)
        # ql = np.array(ql)

        # st = torch.tensor(st,dtype=torch.float)
        # qt = torch.tensor(qt,dtype=torch.float)
        # sl = torch.tensor(sl)
        # ql = torch.tensor(ql)

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
        

    path_val = ['../raw_data/1.npy']

    for p in path_val:
        st, qt, sl, ql = dataloader_with_uni_channels(p,nums_task = config.nums_task_val, time_length=config.time_length)
        # st = np.array(st,dtype=float)
        # qt = np.array(qt,dtype=float)
        # sl = np.array(sl)
        # ql = np.array(ql)

        # st = torch.tensor(st,dtype=torch.float)
        # qt = torch.tensor(qt,dtype=torch.float)
        # sl = torch.tensor(sl)
        # ql = torch.tensor(ql)

    #     sup_train.append(st)
        sup_test += st
        que_test += qt
        sup_label_test += sl
        que_label_test += ql

    # sup_train += sup_test
    # que_train += que_test
    # sup_label += sup_label_test
    # que_label += que_label_test

    sup_train = np.array(sup_train)
    que_train = np.array(que_train)
    sup_test = np.array(sup_test)
    que_test = np.array(que_test)
    np.save("../data/sup_train.npy",sup_train)
    np.save("../data/que_train.npy",que_train)
    np.save("../data/sup_label.npy",sup_label)
    np.save("../data/que_label.npy",que_label)

    np.save("../data/sup_test.npy",sup_test)
    np.save("../data/que_test.npy",que_test)
    np.save("../data/sup_label_test.npy",sup_label_test)
    np.save("../data/que_label_test.npy",que_label_test)

else:
    sup_train = np.load("../data/sup_train.npy", allow_pickle=True)
    que_train = np.load("../data/que_train.npy", allow_pickle=True)
    sup_label = np.load("../data/sup_label.npy", allow_pickle=True)
    que_label = np.load("../data/que_label.npy", allow_pickle=True)

    sup_test = np.load("../data/sup_test.npy", allow_pickle=True)
    que_test = np.load("../data/que_test.npy", allow_pickle=True)
    sup_label_test = np.load("../data/sup_label_test.npy", allow_pickle=True)
    que_label_test = np.load("../data/que_label_test.npy", allow_pickle=True)




# 定义模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Self_Attension_Net(time_length=config.time_length).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=config.lr)
loss_contrastive = Loss()
# animator = d2l.Animator(xlabel='epoch', ylabel='loss',
#                             legend=['train'], xlim=[10, 500])

# loop over the dataset multiple times

# 开始训练
train_list, val_list, train_acc_list, val_acc_list = [],[],[],[]
for epoch in range(config.nums_epoch):
    
#     打乱训练数据
    s = random.random()
    random.seed(s)
    random.shuffle(sup_train)
    random.seed(s)
    random.shuffle(que_train)
    random.seed(s)
    random.shuffle(sup_label)
    random.seed(s)
    random.shuffle(que_label)

    # random.seed(37)
    # random.shuffle(sup_test)
    # random.seed(37)
    # random.shuffle(que_train)
    # random.seed(37)
    # random.shuffle(sup_label)
    # random.seed(37)
    # random.shuffle(que_label)


    
    
    running_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0
    classification_report_train_real = []
    classification_report_train_pred = []
    classification_report_val_real = []
    classification_report_val_pred = []
    
    net.train()
    net.double()
    # for i in range(config.nums_task_train * len(path)):
    n = len(sup_train)
    for i in range(n):
        sup_input = sup_train[i][:,:,:].astype(float)
        sup_input = torch.tensor(sup_input).to(device)
        # sup_input.permute(0,2,1)
        que_input = que_train[i][:,:,:].astype(float)
        que_input = torch.tensor(que_input).to(device)
        # que_input.permute(0,2,1)
        sup_l = sup_label[i][:]
        sup_l = torch.tensor(sup_l).to(device)
        que_l = que_label[i][:]
        que_l = torch.tensor(que_l).to(device)


        optimizer.zero_grad()

        outputs = net(sup_input, que_input)
        loss, pred, label_que, que_pred = loss_contrastive(outputs, (sup_l, que_l))
        if i:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        train_acc += pred.item()

        # label_que.cpu().numpy()
        # classification_report_train_real = np.hstack((classification_report_train_real, label_que))
        # classification_report_train_pred = np.hstack((classification_report_train_pred, que_pred))
            
    # print(f"{epoch} train_loss:{running_loss/config.nums_task_train/len(path)}")
    # train_list.append(running_loss/config.nums_task_train/len(path))
    # print(f" train_acc:{train_acc/config.nums_task_train/len(path)}")
    # train_acc_list.append(train_acc/config.nums_task_train/len(path))
   
    print(f"{epoch} train_loss:{running_loss/n}")
    train_list.append(running_loss/n)
    print(f" train_acc:{train_acc/n}")
    train_acc_list.append(train_acc/n)
    # print(sklearn.metrics.classification_report(classification_report_train_real, classification_report_train_pred))
    
    net.eval()
    with torch.no_grad():
        # for i in range(config.nums_task_val):
        n = len(sup_test)
        for i in range(n):
            # sup_input = sup_train[i][:,:,:].to(device)
            # que_input = que_train[i][:,:,:].to(device)
            # sup_l = sup_label[i][:].to(device)
            # que_l = que_label[i][:].to(device)
            sup_input = sup_test[i][:,:,:].astype(float)
            sup_input = torch.tensor(sup_input).to(device)
            que_input = que_test[i][:,:,:].astype(float)
            que_input = torch.tensor(que_input).to(device)
            sup_l = sup_label_test[i][:]
            sup_l = torch.tensor(sup_l).to(device)
            que_l = que_label_test[i][:]
            que_l = torch.tensor(que_l).to(device)

            outputs = net(sup_input, que_input)
            loss, pred, label_que, que_pred = loss_contrastive(outputs, (sup_l, que_l))
            val_loss += loss.item()
            val_acc += pred.item()

            # label_que.cpu().numpy()
            # # que_pred.cpu().numpy()         
            # classification_report_val_real = np.hstack((classification_report_val_real, label_que))
            # classification_report_val_pred = np.hstack((classification_report_val_pred, que_pred))

            


        print(f"val_loss:{val_loss/n}")
        val_list.append(val_loss/n)
        print(f"val_acc:{val_acc/n}")
        val_acc_list.append(val_acc/n)
        # print(sklearn.metrics.classification_report(classification_report_val_real, classification_report_val_pred))

torch.save(net.state_dict(), "../model/weights.pth")
    
def draw_loss(train_loss, val_loss, train_acc, val_acc, stride=1, path = './'):
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
    
#draw_loss(train_list, val_list,train_acc_list, val_acc_list, stride = 1,path='./')

train_loss = np.array(train_list)
val_loss = np.array(val_list)
train_acc = np.array(train_acc_list)
val_acc = np.array(val_acc_list)

np.save(file="train_loss.npy", arr=train_loss)
np.save(file="val_loss.npy", arr=val_loss)
np.save(file="train_acc.npy", arr=train_acc)
np.save(file="val_acc.npy", arr=val_acc)