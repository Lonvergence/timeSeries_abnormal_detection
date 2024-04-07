import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net import MyNet, Loss
from block import GRUBlock, ConvBlock
import random
import numpy as np
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

data = np.random.randn(10000, 1)
a, b, c, d = process_each_channel(data)
a = np.array(a)
print(a.shape)

