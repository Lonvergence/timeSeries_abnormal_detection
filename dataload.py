import numpy as np
import pandas as pd 
import random

def dataloader(path:str, nums_task = 20,nums_sup=20,nums_que=20,time_length=1000,rate=0.001):
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
            
    sup_train = tasks[:,:nums_sup,:-1,:]
    que_train = tasks[:,nums_sup:,:-1,:]

    sup_label = tasks[:,:nums_sup,-1,:].sum(axis=-1)
    que_label = tasks[:,nums_sup:,-1,:].sum(axis=-1)

    return sup_train, que_train, np.where(sup_label>time_length*rate,1,0), np.where(que_label>time_length*rate,1,0)


# a,b,c,d = dataloader('./pro_data/3_4.npy')
# print(f"==>> d.shape: {d.shape}")
# print(f"==>> c.shape: {c.shape}")
# print(f"==>> b.shape: {b.shape}")
# print(f"==>> a.shape: {a.shape}")



