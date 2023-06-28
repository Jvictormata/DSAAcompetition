import numpy as np
import pandas as pd
import torch
import torch.nn as nn

total_nodes = 836625
data_test = pd.read_csv('test.csv')


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



model = MyNetwork(total_nodes)
model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu'))) 
model.eval()

with torch.no_grad():
    pred = model(torch.tensor(np.array(data_test['id1'])).type(torch.int),torch.tensor(np.array(data_test['id2'])).type(torch.int))
    

submission = pd.read_csv('sample_submission.csv', delimiter=',')
submission['label'] = torch.round(pred)
submission['label'] = submission['label'].astype('int')
submission.to_csv('Predictions2_3.csv',index=False)


