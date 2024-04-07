import numpy as np
import pandas as pd 
import random

def dataloader_with_mul_channels(path:str, nums_task = 20,nums_sup=20,nums_que=20,time_length=1000,rate=0.001):
    data = np.load(path) # [time, channels]
#     data = data.take([0,-1], axis=1)
    mx_time = data.shape[0]

    tasks = [] # [tasks, n_sup + n_que, time_length, channels]

    for _ in range(nums_task):
        task = []
        nums_sample = nums_sup + nums_que

        time_start = random.sample(range(mx_time-time_length+1), nums_sample)

        for begin_idx in time_start: 
                task.append(data[begin_idx:begin_idx+time_length,:])
        
        mean = np.mean(task, axis=(0,1))
        std = np.mean(task, axis=(0,1))
        task = (task - mean) / (std + np.finfo(np.float64).eps)
        
        tasks.append(task)

    tasks = np.array(tasks).transpose(0,1,3,2)
            
    sup_train = tasks[:,:nums_sup,:-1,:] # [tasks, n_sup, channels, time_length]
    que_train = tasks[:,nums_sup:,:-1,:]

    sup_label = tasks[:,:nums_sup,-1,:].sum(axis=-1)
    que_label = tasks[:,nums_sup:,-1,:].sum(axis=-1)

    return sup_train, que_train, np.where(sup_label>time_length*rate,1,0), np.where(que_label>time_length*rate,1,0)


# a,b,c,d = dataloader('./pro_data/3_4.npy')
# print(f"==>> d.shape: {d.shape}")
# print(f"==>> c.shape: {c.shape}")
# print(f"==>> b.shape: {b.shape}")
# print(f"==>> a.shape: {a.shape}")

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

# 将多变量序列拆成单变量数据使用
def dataloader_with_uni_channels(path:str, nums_task = 20,nums_sup=20,nums_que=20,time_length=100):
    data = np.load(path)[:, :-1] # [time, channels]
    assert(len(data.shape) == 2)

    sup_data = []
    que_data = []
    sup_label = []
    que_label = []

    for i in range(data.shape[1]):
        data_uni_channel = data[:][i:i+1]

        sup_data_tmp, # (tasks, n_sup, times_length, 1)
        sup_data_tmp,
        sup_label_tmp, # (tasks, n_sup)
        que_label_tmp = process_each_channel(data_uni_channel, nums_task, nums_sup,nums_que,time_length)

    sup_data += sup_data_tmp
    que_data += sup_data_tmp
    sup_label += sup_label_tmp
    que_label += que_label_tmp


    return sup_data, que_data, sup_label, que_label


