import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

total_nodes = 836625

class MyNetwork(nn.Module):
    def __init__(self, vocabulary_size):
        super(MyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, 100) 
        self.relu = nn.ReLU()
        
        self.lin1_1 = nn.Linear(100, 50)
        self.lin2_1 = nn.Linear(100, 50)
        self.lin1_2 = nn.Linear(50, 25)
        self.lin2_2 = nn.Linear(50, 25)
        
        self.lin_out1 = nn.Linear(50, 25)
        self.lin_out2 = nn.Linear(25, 10)
        self.lin_out3 = nn.Linear(10, 1)
        self.sig = nn.Sigmoid()

        
    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x1 = self.lin1_1(x1)
        x1 = self.relu(x1)
        x1 = self.lin1_2(x1)
        x1 = self.relu(x1)
        
        
        x2 = self.embedding(x2)
        x2 = self.lin2_1(x2)
        x2 = self.relu(x2)
        x2 = self.lin2_2(x2)
        x2 = self.relu(x2)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.lin_out1(x)
        x = self.relu(x)
        x = self.lin_out2(x)
        x = self.relu(x)
        x = self.lin_out3(x)
        x = self.sig(x)
        
        x = x.squeeze()
        return x


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        source_node = self.data.iloc[i]['id1']
        target_node = self.data.iloc[i]['id2']
        link = self.data.iloc[i]['label']
        return source_node, target_node, link


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('train.csv')
train_loader = DataLoader(MyDataset(data), batch_size=1024, shuffle=True)

model = MyNetwork(total_nodes)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


start = time.time()
model.train()
for epoch in range(0, 5):
    for batch_idx, (id1, id2, link) in enumerate(train_loader):
        id1, id2, link = id1.to(device), id2.to(device), link.to(device)
        optimizer.zero_grad()
        output = model(id1.type(torch.int), id2.type(torch.int))

        loss = criterion(output, link.float())
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} {batch_idx * len(id1)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%) - Loss: {loss.item()}')
            
            
print("Elapsed time:")
end = time.time()
print(end - start)


torch.save(model.state_dict(), "model.pt")


