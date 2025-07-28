import torch
import torch.nn as nn
import torch.optim as optim


#Ignore this part
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.Relu = nn.ReLU()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,10)

    def feedforward(self,x):
        x = x.view(-1, 784)
        x = self.Relu(self.fc1(x))
        x = self.Relu(self.fc2(x))
        x = self.fc3(x)
        return x 
  
model = NN()


Cross_Entropy_Loss = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(),lr = 0.01)


epochs = 1000

for epoch in range(epochs):
    running_loss = 0.00
    for images,labels in train_loader:
        optimiser.zero_grad()
        output = model(images)
        loss = Cross_Entropy_Loss(output, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")



