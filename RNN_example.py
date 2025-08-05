import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


char2idx =  {'h': 0, 'e': 1, 'l': 2, 'o': 3, 'p': 4, 'd': 5} # character to index
idx2char = {i: ch for ch, i in char2idx.items()}
input_chars = ['h', 'e', 'l','l']
target_chars = ['e', 'l', 'l','o']
input_indices = []
target_indices = []
for x in input_chars:
     input_indices.append(char2idx[x])
indices = torch.tensor(input_indices)

for x in target_chars:
     target_indices.append(char2idx[x])
target_indices = torch.tensor(target_indices)

one_hot = F.one_hot(indices,num_classes=6).float()



class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_ih = nn.Linear(input_size, hidden_size) # weight's shape: [12,6] ; bias shape: [12]
        self.W_hh = nn.Linear(hidden_size,hidden_size, bias = False) # weight's shape: [12,12] ; no bias
        self.W_ho = nn.Linear(hidden_size, output_size) # weight's shape: [6,12] ; bias shape: [6]
      
       

    def forward(self,x):
        h_t = torch.randn(1,self.hidden_size) #shape: [1,12]
        output = []
        for t in x:
           h_t = torch.tanh(self.W_ih(t) + self.W_hh(h_t)) #shape: [1,12]
           y_t = self.W_ho(h_t)
           output.append(y_t) #shape: [1,6]
        return torch.stack(output) #shape: [4,6]
model = RNN(input_size = 6, hidden_size = 12, output_size = 6)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

epochs = 200
for epoch in range(epochs):
    loss = 0
    output = model(one_hot)
    for i in range(len(output)):
        optimiser.zero_grad()
        loss += criterion(output[i].squeeze(0),target_indices[i])
        # outputs[i]:[1, 6] ; outputs[i].squeeze(0):[6]; target_idx[i]:[]; 
        
    loss.backward()
    optimiser.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")



#check it 
with torch.no_grad():
    outputs = model(one_hot)
    predicted = [torch.argmax(out) for out in outputs]
    predicted_chars = [idx2char[idx.item()] for idx in predicted]
    print("Input:", input_chars)
    print("Predicted next chars:", predicted_chars)




