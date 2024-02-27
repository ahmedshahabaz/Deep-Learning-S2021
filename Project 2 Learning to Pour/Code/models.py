
# you can import pretrained models for experimentation & add your own created models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myRNN(nn.Module):
    

    def __init__(self, args, input_size = 6, direction = 1):
        """
            A linear model for image classification.
        """

        super(myRNN, self).__init__()
        
        self.num_layers = args.lstm_layers
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.direction = direction
        self.sequence = 700
        self.init_dim = self.num_layers * self.direction * self.hidden_size
        self.rnn_type = args.rnn_type

        self.fc_init = nn.Linear(in_features = 6 ,
            out_features = self.init_dim)
        
        self.tanh = nn.Tanh()

        if self.rnn_type.lower() == 'gru':
            self.lstm = nn.GRU(num_layers = self.num_layers, input_size = self.input_size,
                hidden_size = self.hidden_size, batch_first = True, dropout = self.dropout,
                bidirectional = False if direction == 1 else True)
        else:
            self.lstm = nn.LSTM(num_layers = self.num_layers, input_size = self.input_size,
                hidden_size = self.hidden_size, batch_first = True, dropout = self.dropout,
                bidirectional = False if direction == 1 else True)
        
        #self.layer_norm = nn.LayerNorm(self.hidden_size * self.sequence)
        #self.bn = nn.BatchNorm1d(self.sequence * self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * self.direction * self.sequence, self.sequence)
        
        #self.fc1 = nn.Linear(self.sequence * self.hidden_size , self.sequence)


    def forward(self, x, x_lengths, mode = None):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """
        
        x = x.to(torch.float)
        # the observations of the first time step of each data in the batch
        temp = x[:,1,:]
        c0 = self.fc_init(temp)
        c0 = c0.reshape(self.num_layers * self.direction, temp.shape[0],  self.hidden_size)
        h0 = self.tanh(c0)
        x_lengths = x_lengths.reshape(x_lengths.shape[0])

        '''
        https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        '''
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        if self.rnn_type.lower() == 'gru':
            out, hidden = self.lstm(x, h0.detach())
        else:
            out , hidden = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length = 700)
        out = out.view(out.shape[0], self.sequence, self.direction, self.hidden_size)

        # hidden unit for the last time step of the last layer
        #hn = hn.view(self.num_layers, self.direction, out.shape[0], self.hidden_size)[-1]

        '''
        out is the hidden unit for each time step of the last LSTM layer
        hn is the hidden unit for t = 700 of all the layers
        # shape of out is:
        # batch_size, sequence, hidden_size as batch_first = True
        # otherwise sequence, batch_size, hidden_size
        # so out.shape: torch.Size([100, 700, 64])
        # hn.shape-->  batch_size, num_layers, hidden_size
        # so hn.shape: 100, 4, 64
        '''
        out = out.contiguous()
        out = out.view(out.shape[0], self.sequence * self.direction * self.hidden_size)
        #out = self.bn(out)
        
        '''
        else:
            direction_1 = out[:,:,0,:]
            direction_2 = out[:,:,1,:]
            X = torch.cat((direction_1,direction_2), 2)
        '''

        #out = out[:, -1, :]
        # out[:, -1, :] = taking the hidden state of the last time step of all layers
        #out = self.layer_norm(out)
        out = self.fc(out)
        if mode is not None:
            padded_out = out.unsqueeze(dim = -1)
            padded_out = torch.nn.utils.rnn.pack_padded_sequence(padded_out, x_lengths, batch_first=True, enforce_sorted=False)
            padded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(padded_out, batch_first=True, total_length = 700)

            return out, padded_out

        return out
