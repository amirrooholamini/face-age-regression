import argparse
import torch
from torch import nn 
from model import Model
from prepare_dataset import getData
from dataset import Dataset,splitData
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='test model params')
parser.add_argument('--weight_file', type=str, default='face-age-regression.pth', help='model weight file path')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load(args.weight_file, map_location=torch.device(device)))
model.train(False)
model.eval()

width = height = 224
dataset_dir = 'archive/crop_part1'
batch_size = 64

print('preparing data ...')
X,Y = getData(dataset_dir,width,height)
print('data loaded')

dataset = Dataset(X, Y)
test_dataset = splitData(dataset, 0.2, train=False)

dataloader = DataLoader(test_dataset, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

lossFunction = nn.L1Loss()

trainLoss = 0
for images, labels in test_data_loader:
  images = images.to(device)
  labels = labels.to(device)
  predictions = model(images)
  loss = lossFunction(predictions,labels)
  trainLoss += loss
totalLoss = trainLoss / len()
print(totalLoss)