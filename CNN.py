import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

full_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size])

test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)


image, label = train_data[0]
print(label)  # A number from 0 to 9


import matplotlib.pyplot as plt
import matplotlib  # Need to import main package for version
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.show()



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,8,3) #It learns 8 different 3×3 filters
        # Then the output shape is: [batch_size, 8, 26, 26]
        #feature map size [26, 26]
        self.pool = nn.MaxPool2d(2,2) # reduces a 26×26 input to 13 * 13
        self.conv2 = nn.Conv2d(8,16,3)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128,10)

    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
model = CNN()

def accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in dataloader:
            output = model(image)
            _, predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    model.train()
    return 100 * correct / total

Cross_Entropy_Loss = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

epochs = 10000
    
for epoch in range(epochs):
    running_loss = 0
    for image, label in train_loader:
        optimiser.zero_grad()
        output = model(image)
        loss = Cross_Entropy_Loss(output, label)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
   
    
    # Accuracy evaluation
train_acc = accuracy(model, train_loader)
val_acc   = accuracy(model, val_loader)
print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

test_acc = accuracy(model, test_loader)
print(f"\n Final Test Accuracy: {test_acc:.2f}%")






    
    
