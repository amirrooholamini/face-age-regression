import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, p = 0.2):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, (3,3) , (1,1), (1,1)) # input channel - output channel - mask size , stride - padding
    self.conv2 = nn.Conv2d(32, 64, (3,3) , (1,1), (1,1))
    self.conv3 = nn.Conv2d(64, 128, (3,3) , (1,1), (1,1))
    self.conv4 = nn.Conv2d(128, 256, (3,3) , (1,1), (1,1))
    self.conv5 = nn.Conv2d(256, 512, (3,3) , (1,1), (1,1))

    self.fc1 = nn.Linear(25088, 256)
    self.fc2 = nn.Linear(256, 64)
    self.fc3 = nn.Linear(64, 1)
    self.dropout = nn.Dropout(p) 

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, kernel_size=(2,2))

    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, kernel_size=(2,2))

    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, kernel_size=(2,2))

    x = F.relu(self.conv4(x))
    x = F.max_pool2d(x, kernel_size=(2,2))

    x = F.relu(self.conv5(x))
    x = F.max_pool2d(x, kernel_size=(2,2))
   
    
    x = torch.flatten(x, start_dim=1)
    self.dropout(x)
    x = self.fc1(x)
    self.dropout(x)
    x = self.fc2(x)
    self.dropout(x)
    x = self.fc3(x)


    return x
