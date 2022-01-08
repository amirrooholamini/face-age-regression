from prepare_dataset import getData
from dataset import Dataset,splitData

from torch.utils.data import DataLoader
from model import Model
import torch
from torch import nn
from tqdm import tqdm


import argparse
parser = argparse.ArgumentParser(description='train model params')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset_dir', type=str, default='archive/crop_part1')
args = parser.parse_args()

width = height = args.image_size
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
dataset_dir = args.dataset_dir

print('preparing data ...')
X,Y = getData(dataset_dir,width,height)
print('data loaded')

dataset = Dataset(X, Y)
train_dataset = splitData(dataset, 0.2, True)

dataloader = DataLoader(train_dataset, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossFunction = nn.L1Loss()


for epoch in range(epochs):
  trainLoss = 0
  for images, labels in tqdm(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    predictions = model(images)
    labels = labels.to(torch.float32)
    predictions = predictions.to(torch.float32)
    loss = lossFunction(predictions,labels)
    loss.backward()
    optimizer.step()

    trainLoss += loss

  totalLoss = trainLoss/len(dataloader)
  print(f"Epoch: {epoch+1} - Loss: {totalLoss}")

torch.save(model.state_dict(), "face-age-regression.pth")