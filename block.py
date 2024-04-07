import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvBlock(nn.Module):
    """Some Information about ConvBlock"""
    def __init__(self, in_channels = 1, out_channels = 32, mid_channels = None,time_length = 100):
        super(ConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_lenth = time_length
        if mid_channels == None:
            self.mid_channels = out_channels
        else:
            self.mid_channels = mid_channels

        self.net = nn.Sequential(
            nn.Conv1d(self.in_channels, self.mid_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_channels, self.out_channels, kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_data):

        # # input_data  [batch_size, in_channels, time_length]
        # shape = input_data.shape
        # N = shape[0]
        # C = shape[1]   # C = 1
        # T = shape[2]
        
        # x = input_data
        # if len(shape) == 3:
        #     x = torch.unsqueeze(input_data, -1).permute(0,1,3,2) # x [N, C, 1, T]
        # elif len(shape) == 4:
        #     x = x.permute(0,1,3,2)

        # x = x.view(N * C, self.in_channels, T)

        # y = self.net(x) # y [N * C, 32, T]

        # output = y.view(N, C, self.out_channels, T) # output [N, C, K, T]

        # output = torch.mean(output, dim=1) # output [N, K, T]

        # return output

        shape = input_data.shape
        N = shape[0]
        C = shape[1]   # C = 1
        T = shape[2]

        y = self.net(input_data)

        return y



class GRUBlock(nn.Module):
    """Some Information about GRUBlock"""
    def __init__(self,input_size = 1, hidden_size = 32):
        super(GRUBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # net
        self.gru1 = nn.GRU(input_size, hidden_size, 1, batch_first = True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first = True)
        self.gru3 = nn.GRU(hidden_size, hidden_size, 1, batch_first = True)

    def forward(self, x):
        # x [batch_size, nums_channels, time_length] [N, C, T]

        shape = x.shape
        N = shape[0]
        C = shape[1]
        T = shape[2]

        x = torch.unsqueeze(x, -1) # [N, C, T, 1]
        x = x.view(N * C, T, 1)

        y, _ = self.gru1(x) 
        y, _ = self.gru2(y)
        output, _ = self.gru3(y) # output [T, N * C, K=32]

        output = output.permute(1, 0, 2).view(N, C, T, self.hidden_size) # ouput [N, C, T, K]
        
#         output = torch.mean(output, dim=1).permute(0,2,1)
        
        return output