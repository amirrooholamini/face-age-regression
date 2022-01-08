import argparse

import time
start = time.time()
import torch
import torchvision
import cv2 as cv
import numpy as np
from model import Model
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='inference model params')
parser.add_argument('--image', type=str, default='images/face.jpg', help='image path')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--weights', type=str, default='face-age-regression.pth')
args = parser.parse_args()

inference_transform = torchvision.transforms.Compose([
     torchvision.transforms.ToPILImage(mode=None),
     torchvision.transforms.Resize((224,224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0),(1)),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load(args.weights, map_location ='cpu'))

model.train(False)
model.eval()

img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img,(args.image_size,args.image_size))
tensor = inference_transform(img).unsqueeze(0).to(device)
prediction = model(tensor).cpu().detach().numpy()
print(np.argmax(prediction, axis=1))
end = time.time()
print(f'time: {end-start} seconds')