import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Example sentence dataset(prepare the data)
sentences = [
    "I love to eat",
    "I love to code",
    "You love to learn",
    "You love to eat",
    "They love to play"
]


tokenised_words = set()
for sentence in sentences:
    words = sentence.lower().split()
    for word in words:
        tokenised_words.add(word)


word2idx = {}
index = 0
for word in tokenised_words:
    word2idx[word] = index
    index += 1

idx2word = {}
index = 0
for word in tokenised_words:
    idx2word[index] = word
    index += 1


#create training data
data = []
for sentence in sentences:
    words = sentence.lower().split()
    for i in range(len(words) - 1):
        input_seq = words[:i+1]
        target = words[i+1]
        data.append((torch.tensor([word2idx[w] for w in input_seq]), 
                                 torch.tensor(word2idx[target])
                    ))

#print(data)
#define the RNN model

class RNN_LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
       super(RNN_LM,self).__init__()
       self.embedding = nn.Embedding(vocab_size, embedding_dim)
       self.rnn = nn.RNN(embedding_dim,hidden_dim, batch_first= True)
       self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
         x = self.embedding(x)
         out, _ = self.rnn(x)
         out = self.fc(out[:,-1,:])
         return out


model = RNN_LM(vocab_size = len(tokenised_words), embedding_dim=16, hidden_dim=32)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr = 0.01)

epochs = 200
for epoch in range(epochs):
    total_loss = 0
    for input_seq, target in data:
        optimiser.zero_grad()
        input_seq = input_seq.unsqueeze(0)
        target = target.unsqueeze(0)
        output = model(input_seq)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
        total_loss += loss
        

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")


#I will figure it our sometime later
def predict_next_word(model, input_words):
    model.eval()
    input_ids = torch.tensor([word2idx[w] for w in input_words.lower().split()])
    input_ids = input_ids.unsqueeze(0)  # Add batch dimension
    output = model(input_ids)
    pred_idx = torch.argmax(output, dim=1).item()
    return idx2word[pred_idx]

# Test prediction
print(predict_next_word(model, "I love to"))  # Should return: eat or code
