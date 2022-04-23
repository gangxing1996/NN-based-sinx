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
model=net();
model_info = torch.load('./pa.pt')
model.load_state_dict(model_info);
model.eval();
#print (torch.tensor(float(input())).float().squ)
print(model(torch.tensor(float(input())).float().unsqueeze(-1)));

