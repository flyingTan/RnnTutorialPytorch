import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


#1. Generate data for Training the network
T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int32')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float32')

#2. Split data into train and test
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
input = torch.from_numpy(data[3:, :-1]).to(device)
target = torch.from_numpy(data[3:, 1:]).to(device)
test_input = torch.from_numpy(data[:3, :-1]).to(device)
test_target = torch.from_numpy(data[:3, 1:]).to(device)

import torch.nn as nn

class RnnCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RnnCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.act = torch.nn.Tanh()
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.act(hidden)
        return hidden

class Sequence(nn.Module):
    def __init__(self, input_size,
                 rnn_num: int = 1, 
                 hidden_size: list = [128], 
                 output_size: int = 1, 
                 use_customer_rnn = True):
        r""" RNNs based on RnnCell
        Args:
            input_size: Embedding feature
            rnn_num: The number of rnncell
            hidden_size: The hidden size of all RnnCells
            ouput_size: Output size of FC
            use_customer_rnn: use customer rnn 
        """
        super(Sequence, self).__init__()
        assert(rnn_num == len(hidden_size))
        self.input_size = input_size
        self.rnn_num = rnn_num
        self.hidden_size = hidden_size
        self.use_customer_rnn = use_customer_rnn
        rnn_type = RnnCell if self.use_customer_rnn else torch.nn.RNNCell
        rnn_names = self.__dict__
        for idx in range(self.rnn_num):
            if idx == 0:
                setattr(self, 'rnn{}'.format(idx), rnn_type(input_size, hidden_size[idx]))
            else:
                setattr(self, 'rnn{}'.format(idx), rnn_type(hidden_size[idx - 1], hidden_size[idx]))
            rnn_names['rnn_hidden{}_size'.format(idx)] = self.hidden_size[idx]
        self.fc1  = torch.nn.Linear(self.hidden_size[-1], output_size)
    
    def init_hidden(self, batch_size):
        hidden_init_list = []
        for i in range(self.rnn_num):
            hidden_init_list.append(torch.zeros(batch_size, self.hidden_size[i]).to(device))
        return hidden_init_list
        
    def forward(self, input:torch.tensor, feature_n = 0):
        """ foward of sequence impl by tanfeiyang
        Agrs:
            input: (Time * Batchsize * Feature_embeding) tensor.
            feature_n: ( N ) Predict next n status
        Reture:
            output: (T * B * F) tensor
        """
        hidden_val = self.init_hidden(input.shape[1])
        
        outputs = []
        for time_clip in range(input.shape[0]):
            batch_data = input[time_clip].to(device)
            for idx in range(self.rnn_num):
                if idx == 0:
                    hidden_val[idx] = getattr(self, "rnn{}".format(idx))(batch_data, hidden_val[idx])
                else:
                    hidden_val[idx] = getattr(self, "rnn{}".format(idx))(hidden_val[idx - 1], hidden_val[idx])
            ouput = self.fc1(hidden_val[-1])
            outputs.append(ouput)

        """ Predict next n status """
        for t_i in range(feature_n):
            for idx in range(self.rnn_num):
                if idx == 0:
                    hidden_val[idx] = getattr(self, "rnn{}".format(idx))(outputs[-1], hidden_val[idx])
                else:
                    hidden_val[idx] = getattr(self, "rnn{}".format(idx))(hidden_val[idx - 1], hidden_val[idx])
            ouput = self.fc1(hidden_val[-1])
            outputs.append(ouput) 

        #outputs = torch.stack(outputs, 0).squeeze().permute(1, 0)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

#seq = Sequence(1, 2, [100, 200], 1, False)
seq = Sequence(1, 1, [100], 1, False)
seq = seq.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(seq.parameters(), lr=0.1, momentum=0.9)

with torch.autograd.set_detect_anomaly(False):
    for i in range(100):
        for _ in  range(30):
            optimizer.zero_grad()
            input_data = input.reshape((97, 999, 1)).permute(1, 0, 2)
            outputs = seq(input_data)
            loss = criterion(outputs, target)
            print('STEP:', i,'loss:', loss.item(), '\n')
            loss.backward()
            optimizer.step()
    
        future = 1000
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            input_test_data = test_input.reshape((3, 999, 1)).permute(1, 0, 2)
            output_test = seq(input_test_data, future)
            loss = criterion(output_test[:, :-future], test_target)
            print(' test loss:', loss.item(), '\n')
            preds = output_test.detach().cpu().numpy()
        ## Draw
        sns.set_theme()
        fig, axes = plt.subplots(1, figsize=(15, 5))
        for ith_curve in range(preds.shape[0]):
            axes.plot(preds[ith_curve][0:], label='curve_{}'.format(ith_curve))
        plt.legend()
        fig.savefig('./show/test_sin_{}.jpg'.format(str(i)))
        
