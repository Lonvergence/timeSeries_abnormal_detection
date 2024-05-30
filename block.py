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
            # nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=3,padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(self.mid_channels, self.mid_channels, kernel_size=3,padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(self.mid_channels, self.out_channels, kernel_size=3,padding=1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, input_data):
        # # input_data  [batch_size, in_channels, time_length]
        # shape = input_data.shape
        # N = shape[0]
        # C = shape[1]   
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
        # x = input_data.permute(0, 2, 1)
        x = input_data
        shape = x.shape
        N = shape[0]
        C = shape[1]   # C = 1
        T = shape[2]

        y = self.net(x)

        return y



class Conv_LSTM(nn.Module):

    def __init__(self,input_size = 1, hidden_size = 32):
        super(Conv_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # net
        self.conv1 = nn.Conv1d(input_size, 16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv1d(16, 16,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv1d(16, 16,kernel_size=3,stride=2,padding=1)
        self.gru1 = nn.LSTM(16, hidden_size, 1, batch_first = True)
        # self.gru2 = nn.GRU(hidden_size, hidden_size, 1, batch_first = True)
        # self.gru3 = nn.GRU(hidden_size, hidden_size, 1, batch_first = True)

    def forward(self, x):
        # x [batch_size, nums_channels, time_length] [N, C, T]
        # shape = x.shape
        # N = shape[0]
        # C = shape[1]
        # T = shape[2]

        # x = torch.unsqueeze(x, -1) # [N, C, T, 1]
        # x = x.view(N * C, T, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        output, _ = self.gru1(x) 

        # y, _ = self.gru2(y)
        # output, _ = self.gru3(y) # output [T, N * C, K=32]

        # output = output.permute(1, 0, 2).view(N, C, T, self.hidden_size) # ouput [N, C, T, K]
        
#         output = torch.mean(output, dim=1).permute(0,2,1)
        
        return output.permute(0, 2, 1)
    

class ResBlock(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,time_length,
                 use_1x1conv=False, strides=1):
        super().__init__()

        self.time_length = time_length

        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.ln1 = nn.LayerNorm(self.time_length)
        self.ln2 = nn.LayerNorm(self.time_length)

    def forward(self, X):
        # X = X.permute(0, 2, 1)
        Y = F.relu(self.ln1(self.conv1(X)))
        Y = self.ln2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, length = 1000):
        super(SelfAttention, self).__init__()
        self.length = length

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.ln = nn.LayerNorm(self.length)

    def forward(self, x):
        x = x.permute(0,2,1)
        origin = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.permute(0, 2, 1))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        attended_values += origin
        attended_values = attended_values.permute(0, 2, 1)
        attended_values = self.ln(attended_values)
        return attended_values