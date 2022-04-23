import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__();
        self.fc1=nn.Linear(1,10);
        self.fc2=nn.Linear(10,20);
        self.fc3=nn.Linear(20,1);
        
    def forward(self,x):
        x=self.fc1(x);
        x=F.tanh(x);
        x=self.fc2(x);
        x=F.tanh(x);
        x=self.fc3(x);
        x=F.tanh(x);
        return x;
class sindataset(data.Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath,mode="r") as f:
            for line in f:
                line = line.strip()
                elem = line.split()
                x = float(elem[0])
                y = float(elem[1])
                #print(f'sindataset.__init__ "{x}, {y}"')
                self.data.append((x,y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
    
    
model=net()
optimizer = optim.Adam(model.parameters(), lr=0.01);
ceriation = nn.MSELoss();

thedataset = sindataset('./data.txt')

dataloader = data.DataLoader(thedataset, batch_size=128, shuffle=True)

model.train()
for epoch in range(100):
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad();
        x, y = batch
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        x = x.float()
        y = y.float()
        #import ipdb; ipdb.set_trace()
        out = model(x)
        loss=ceriation(out,y);
        print(loss.item())
        loss.backward();
        optimizer.step();

torch.save(model.state_dict(),"pa.pt");
        
           
        