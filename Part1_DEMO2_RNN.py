import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch.nn as nn

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

class Sequence(nn.Module):
    def __init__(self, input_feature_size, rnn_num, hidden_size, output_size):
        super(Sequence, self).__init__()
        self.rnn = nn.RNN(input_feature_size,
                          hidden_size, 
                          rnn_num,
                          batch_first= False)
        self.fc1 = nn.Linear(hidden_size, output_size)
    
    def init_hidden(self,rnn_num, directions = 1, batch_size = 128, hidden_size = 128, device = 'cpu'):
        return torch.zeros(rnn_num * directions, batch_size, hidden_size).to(device)
    
    def forward(self, input: torch.tensor, hidden: torch.tensor, future_n = 0):
        r""" Convert Input: Time * Batchsize * Featurelength formation """
        input_data = input.reshape((input.shape[0], input.shape[1], 1)).permute(1, 0, 2)

        output, hidden = self.rnn(input_data, hidden)
        outputs = []
        for time_clip in range(input_data.shape[0]):
            outputs.append(self.fc1(output[time_clip, :, :]))
        
        """ Predict next n status """
        for t_i in range(future_n):
            output_future, hidden = self.rnn(outputs[-1].expand(1,-1,1), hidden)
            outputs.append(self.fc1(output_future.squeeze()))
        
        outputs = torch.stack(outputs, 0).squeeze().permute(1, 0)
        return outputs

#seq = Sequence(1, 2, [100, 200], 1, False)
seq = Sequence(1, 2, 100, 1)
seq = seq.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(seq.parameters(), lr=0.1, momentum=0.9)

with torch.autograd.set_detect_anomaly(False):
    for i in range(100):
        for _ in  range(30):
            optimizer.zero_grad()
            outputs = seq(input, seq.init_hidden(2, 1, input.shape[0], 100, device))
            loss = criterion(outputs, target)
            print('STEP:', i,'loss:', loss.item(), '\n')
            loss.backward()
            optimizer.step()
    
        future = 1000
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            output_test = seq(test_input, 
                              seq.init_hidden(2, 1, test_input.shape[0], 100, device),
                              future_n = 1000)
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