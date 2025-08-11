import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM_Cell(nn.Module):
    #This is a single time step logic of the GRU.
    def __init__(self, input_size, hidden_size):
        super(LSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #forget gate parameters
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)

        #update input parameters
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
    
        #output state
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

        #candidate state
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

    
    def forward(self, x_t, state):
        a_prev, c_prev = state
        forget_gate = torch.sigmoid(self.W_f(x_t) + self.U_f(a_prev))
        input_gate = torch.sigmoid(self.W_i(x_t) + self.U_i(a_prev))
        output_gate = torch.sigmoid(self.W_o(x_t) + self.U_o(a_prev))
        g_t = torch.tanh(self.W_c(x_t) + self.U_c(a_prev))
        c_t =  forget_gate * c_prev + input_gate * g_t
        a_t = output_gate * torch.tanh(c_t)
      
        
        return a_t, c_t
    

class Deep_Bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers = 3, dropout = 0.2):
        super().__init__()

        self.LSTM_Cell = nn.LSTM(input_size = input_size , hidden_size = hidden_size, num_layers = num_layers, bidirectional = True, batch_first = True, dropout = dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size*2 , num_classes)
    
    def forward(self, inputs):
        #Input x: (batch_size, seq_len, feature dimension at each time step)  # batch_first=True
        #Output: (batch, num_classes)
        _, (a_t, _)   = self.lstm(inputs)
        h_last_forward = a_t[-2]
        h_last_backward = a_t[-1]
        h = torch.cat([h_last_forward, h_last_backward], dim =1)
        return self.fc(h)