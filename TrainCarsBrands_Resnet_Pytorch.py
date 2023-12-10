# adapted and simplified to consider 49 brands or makes instead 196 models
# https://www.kaggle.com/code/hussnain47/car-object-detection-and-classification
# By Alfonso Blanco García, December 2023

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob
import io
import os
import cv2
#import json
#import shutil
#import numpy as np
#import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image



train_transforms = transforms.Compose([transforms.Resize((244,244)),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# The validation set will use the same transform as the test set
test_transforms = transforms.Compose([transforms.Resize((244,244)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validation_transforms = transforms.Compose([transforms.Resize((244,244)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_data = datasets.ImageFolder( 'KaggleCarsByBrands_1_49/train', transform=train_transforms)
test_data = datasets.ImageFolder( 'KaggleCarsByBrands_1_49/valid', transform=test_transforms)

train_data, valid_data = torch.utils.data.random_split(train_data, [0.7, 0.3])

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

#from torchvision.models import ResNet50_Weights
model = models.resnet50(pretrained=True)
#model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 196)
model.fc = nn.Linear(num_ftrs, 49)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    # change model to work with cuda
    #model.to('cuda')
    # change model to work with cpu
    model.to('cpu')

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):
    
        # Change images and labels to work with cuda
        #images, labels = images.to('cuda'), labels.to('cuda')
        # Change images and labels to work with cpu
        images, labels = images.to('cpu'), labels.to('cpu')

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

#device = torch.device("cuda")
device = torch.device("cpu")
model = model.to(device)

epochs = 20
steps = 0
print_every = 40

# change to gpu mode
# model.to('cuda')
# change to cpu mode
model.to('cpu')
model.train()
for e in range(epochs):

    running_loss = 0
    
    # Iterating over data to carry out training step
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        #inputs, labels = inputs.to('cuda'), labels.to('cuda')
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        
        # zeroing parameter gradients
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Carrying out validation step
        if steps % print_every == 0:
            # setting model to evaluation mode during validation
            model.eval()
            
            # Gradients are turned off as no longer in training
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
            
            print(f"No. epochs: {e+1}, \
            Training Loss: {round(running_loss/print_every,3)} \
            Valid Loss: {round(valid_loss/len(validloader),3)} \
            Valid Accuracy: {round(float(accuracy/len(validloader)),3)}")
            
            
            # Turning training back on
            model.train()
            lrscheduler.step(accuracy * 100)


correct = 0
total = 0
#model.to('cuda')
model.to('cpu')


with torch.no_grad():
    for data in testloader:
        images, labels = data
        #images, labels = images.to('cuda'), labels.to('cuda')
        images, labels = images.to('cpu'), labels.to('cpu')
        # Get probabilities
        outputs = model(images)
        # Turn probabilities into predictions
        _, predicted_outcome = torch.max(outputs.data, 1)
        # Total number of images
        total += labels.size(0)
        # Count number of cases in which predictions are correct
        correct += (predicted_outcome == labels).sum().item()

print(f"Test accuracy of model: {round(100 * correct / total,3)}%")

checkpoint = {'state_dict': model.state_dict(),
              'model': model.fc,
              #'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict,
              'num_epochs': epochs}


torch.save(checkpoint, 'checkpoint20epoch.pth')
