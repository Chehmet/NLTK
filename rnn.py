import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  
        return out

input_size = 1
hidden_size = 16
output_size = 1
num_layers = 2

model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = torch.tensor([[10, 20, 15, 25, 30, 35, 40]], dtype=torch.float32).view(-1, 7, 1)
target = torch.tensor([45], dtype=torch.float32).view(-1, 1)

for epoch in range(200):
    model.train()
    output = model(data)
    loss = criterion(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
predicted = model(data)
print("Predicted visitors for the next day:", predicted.item())
