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

def process_each_channel(data, nums_task = 20,nums_sup = 24,nums_que = 24,time_length = 100):
    # data 单变量时间序列
    sup_data_return = []
    que_data_return = []
    sup_label = []
    que_label = []

    mx_time = data.shape[0]
    
    for _ in range(nums_task):
        sup_label_tmp = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4
        que_label_tmp = [0] * 8 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4
        sup_data_tmp = []
        que_data_tmp = []

        # time_start_list = random.sample(range(mx_time-time_length+1), nums_sup + nums_que)
        nums_sample = nums_sup + nums_que
        time_s = random.randint(0, mx_time - 1 - time_length - (nums_sample - 1) * time_length // 100);
        stop = time_s + (nums_sample) * time_length // 100
        time_start_list = range(time_s, stop, time_length // 100)
        time_start_list = list(time_start_list)
        random.shuffle(time_start_list)
        data_sample = [data[i:i + time_length] for i in time_start_list]

        mean = np.mean(data_sample, axis=(0,1))
        std = np.mean(data_sample, axis=(0,1))
        data_sample = (data_sample - mean) / (std + np.finfo(np.float64).eps)

        sup_data_tmp = data_sample[0:nums_sup]
        que_data_tmp = data_sample[nums_sup:nums_que + nums_sup]

        sup_data = []
        que_data = []
        for i, label in enumerate(sup_label_tmp):
            # x, y = generate_sequence(0, time_length)
            x = time_length // 2 - random.randint(1000, 10000)
            y = time_length // 2 + random.randint(1000, 10000)
            if (y - x) % 2:
                y -= 1
            
            if label == 0:
                pass
            elif label == 1:
                ab = max(sup_data_tmp[i])
                sup_data_tmp[i][x:y] = ab
            elif label == 2:
                ab = min(sup_data_tmp[i])
                sup_data_tmp[i][x:y] = ab
            elif label == 3:
                sup_data_tmp[i][x:int((x + y) / 2)] = sup_data_tmp[i][int((x + y) // 2):y] 
                # sup_data_tmp[i][0:time_length // 2] = sup_data_tmp[i][time_length//2 :]
            else: 
                sup_data_tmp[i][0:time_length // 2] = sup_data_tmp[i][time_length//2 :]
                sup_data_tmp[i][int((x + y) // 2):y] = sup_data_tmp[i][x:int((x + y) // 2)] 
            tmp = sup_data_tmp[i].reshape((-1,))
            tmp = tmp.reshape((-1,20)).mean(axis=1)
            tmp = tmp.reshape((-1,1))
            sup_data.append(tmp)



        for i, label in enumerate(que_label_tmp):
            x = time_length // 2 - random.randint(1000, 10000)
            y = time_length // 2 + random.randint(1000, 10000)
            if (y - x) % 2:
                y -= 1
            if label == 0:
                pass
            elif label == 1:
                ab = max(que_data_tmp[i])
                que_data_tmp[i][x:y] = ab
            elif label == 2:
                ab = min(que_data_tmp[i])
                que_data_tmp[i][x:y] = ab
            elif label == 3:
                que_data_tmp[i][x:int((x + y) // 2)] = que_data_tmp[i][int((x + y) // 2):y]
                # que_data_tmp[i][0:time_length//2] = que_data_tmp[i][time_length // 2:] 
            else:
                que_data_tmp[i][0:time_length//2] = que_data_tmp[i][time_length // 2:] 
                que_data_tmp[i][int((x + y) // 2):y] = que_data_tmp[i][x:int((x + y) // 2)]
            tmp = que_data_tmp[i].reshape((-1,))
            tmp = tmp.reshape((-1,20)).mean(axis=1)
            tmp = tmp.reshape((-1,1))
            que_data.append(tmp)
        
        sup_data_return.append(sup_data)
        que_data_return.append(que_data)
        sup_label.append(sup_label_tmp)
        que_label.append(que_label_tmp)


    return sup_data_return, que_data_return, sup_label, que_label

# 将多变量序列拆成单变量数据使用
def dataloader_with_uni_channels(path:str, nums_task = 20,nums_sup=24,nums_que=24,time_length=100):
    data = np.load(path)[:, :] # [time, channels]

    assert(len(data.shape) == 2)

    sup_data = []
    que_data = []
    sup_label = []
    que_label = []

    # for i in range(data.shape[1]):
    for i in range(1):
        data_uni_channel = data[:,i:i+1]
        # (tasks, n_sup, times_length, 1)
        sup_data_tmp, sup_data_tmp, sup_label_tmp, que_label_tmp = process_each_channel(data_uni_channel, nums_task, nums_sup,nums_que,time_length)

        sup_data += sup_data_tmp
        que_data += sup_data_tmp
        sup_label += sup_label_tmp
        que_label += que_label_tmp


    return sup_data, que_data, sup_label, que_label


def generate_sequence(mn: int, mx: int):
    x = random.randint(mn, mx)
    y = random.randint(mn, mx)
    if x > y: 
        x, y = y, x
    # y > x
    if x == y or x == mn or y == mx:
        return mn, mx
    
    if (y - x) % 2 == 1:
        y += 1
    return x, y
    