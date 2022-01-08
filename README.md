# Assignment 2 - Part 2 (Face Age Regression)

train a model on persian mnist dataset

## Installation
pull package in your system.
Then use [pip](https://pip.pypa.io/en/stable/) to install packages

```bash
pip install torch
pip install torchvision
pip install opencv
```

Or run requirments.txt file:
```bash
pip install -r requirements.txt
```
## Train

run the 
[train.py](https://pip.pypa.io/en/stable/) file to train model and save weights. you can send parameters as arguments to this file

```bash
python3 train.py --epochs [EPOCHS] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --image_size [IMAGE_SIZE] --dataset_dir [DATASET_DIR]
```


## Test
after train model to can test your model on test data in fashion mnist dataset by run [test.py](https://pip.pypa.io/en/stable/) file

```bash
python3 test.py --weights [weights]
```

## Inference
after train and test model you can send a sample image to [inference.py](https://pip.pypa.io/en/stable/) file to predict image.

```bash
python3 test.py --image [IMAGE_PATH]
```