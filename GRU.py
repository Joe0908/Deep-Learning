import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GRU_Cell(nn.Module):
    #This is a single time step logic of the GRU.
    def __init__(self, input_size, hidden_size):
        super(GRU_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        #update gate parameters
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
    
        #candidate hidden state
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

    
    def forward(self, x_t, h_prev):
        reset_gate = nn.Sigmoid(self.W_r(x_t) + self.U_r(h_prev))
        update_gate = nn.Sigmoid(self.W_u(x_t) + self.U_u(h_prev))
        candidate_hidden = nn.Tanh(self.W(x_t) + self.U(h_prev * reset_gate))
        final_hidden = (1-update_gate) * h_prev + update_gate * candidate_hidden
        
        return final_hidden
    

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomGRU,self).__init__()
        self.hidden_size = hidden_size
        self.GRU_Cell = GRU_Cell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size , output_size)
    
    def forward(self, inputs):
        # inputs: (seq_len, input_size)
        # If you have a 3-word sequence and each word is a 5-dim vector:
        # inputs.shape == (3, 5)
        h_t = torch.zeros(self.hidden_size)
        #init hidden state
        for x_t in inputs:
            h_t = self.GRU_Cell(x_t, h_t)
        output = self.fc(h_t)
        return output



# Hyperparameters
input_size = 5
hidden_size = 10
output_size = 2
learning_rate = 0.01
epochs = 100

# Model
model = CustomGRU(input_size, hidden_size, output_size)
Criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

# Dummy data (e.g., sequence of 3 one-hot inputs)
x_seq = [torch.eye(input_size)[torch.randint(0, input_size, (1,)).item()] for _ in range(3)]
x_seq = torch.stack(x_seq)  # shape: [3, 5]

target_class = torch.tensor([1])  # random label

for epoch in range(epochs):
    optimiser.zero_grad()
    output = model(x_seq)
    loss = Criterion(output.unsqueeze(0, target_class))
    loss.backward()
    optimiser.step()
    
    if (epoch + 1) % 10 == 0:
        pred = torch.argmax(output).item()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Prediction = {pred}")





