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
    

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.LSTM_Cell = LSTM_Cell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size , output_size)
    
    def forward(self, inputs):
        # inputs: (seq_len, input_size)
        # If you have a 3-word sequence and each word is a 5-dim vector:
        # inputs.shape == (3, 5)
        a_t = torch.zeros(self.hidden_size)
        c_t = torch.zeros(self.hidden_size)
        #init hidden state
        for x_t in inputs:
            a_t, c_t= self.LSTM_Cell(x_t, (a_t,c_t))
        output = self.fc(a_t)
        return output



# Hyperparameters
input_size = 5
hidden_size = 10
output_size = 2
learning_rate = 0.01
epochs = 100

# Model
model = CustomLSTM(input_size, hidden_size, output_size)
Criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

# Dummy data (e.g., sequence of 3 one-hot inputs)
x_seq = [torch.eye(input_size)[torch.randint(0, input_size, (1,)).item()] for _ in range(3)]
x_seq = torch.stack(x_seq)  # shape: [3, 5]

target_class = torch.tensor([1])  # random label

for epoch in range(epochs):
    optimiser.zero_grad()
    output = model(x_seq)
    loss = Criterion(output.unsqueeze(0), target_class)
    loss.backward()
    optimiser.step()
    
    if (epoch + 1) % 10 == 0:
        pred = torch.argmax(output).item()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Prediction = {pred}")





