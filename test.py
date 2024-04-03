import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net import MyNet, Loss
from block import GRUBlock, ConvBlock

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








