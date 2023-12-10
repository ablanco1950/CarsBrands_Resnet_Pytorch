# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , Jun 2023
"""

######################################################################
# PARAMETERS
######################################################################

######################################################################
import torch
from torch import nn
import os
import re

import cv2

import numpy as np
import keras
import functools  
import time
inicio=time.time()

from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image

#model = models.resnet34(pretrained=True)
model = models.resnet50(pretrained=True)

# https://stackoverflow.com/questions/53612835/size-mismatch-for-fc-bias-and-fc-weight-in-pytorch
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 49)

TabCarBrand=[]
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#model_path= "my_checkpoint1.pth"
model_path= "checkpoint20epoch.pth"

#model = load_checkpoint('/kaggle/working/my_checkpoint1.pth')
#model = load_checkpoint('my_checkpoint1.pth')
model = load_checkpoint('checkpoint20epoch.pth')
# Checking model i.e. should have 196 output units in the classifier
#print(model)
DataPath='C:\\archiveKaggle\\cars_train\\cars_train' + '\\'

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
#classes, c_to_idx = find_classes(data_dir+"/train")
classes, c_to_idx = find_classes('KaggleCarsByBrands_1_49/train')


print(classes, c_to_idx)

f=open("CarBrand.csv","r")

for linea in f:
    lineadelTrain =linea.split(",")
    TabCarBrand.append(lineadelTrain[3])

def GetBrandFromModel(Modelo):
    f=open("CarBrand.csv","r")
    
    for linea in f:
        lineadelTrain =linea.split(",")
        ModelFrom=int(lineadelTrain[1])
        ModelTo=int(lineadelTrain[2])
        if Modelo >= ModelFrom and Modelo <= ModelTo:
            return int(lineadelTrain[0]), lineadelTrain[3]
    print("RARO NO ENCUENTRA EL MODELO")
    return -1, ""

def loadimagesTest():
    
    #images=[]
    TabImagePath=[]
    Y=[]
    imagesName=[]
    f=open("cardatasettrain.csv","r")
    ContTraining=0
    ContValid=0
    Conta=0;
    for linea in f:
        Conta=Conta+1
        if Conta==1: continue
        
        
        if Conta < 8000: continue
        
        if Conta > 8135: break # van algunas en blanco y negro que el sistema no acepta
       
        lineadelTrain =linea.split(",")
       
        NameImg=lineadelTrain[6]
        # OJO LLEVA UN CR AL FINAL
        NameImg=NameImg[0:9]
        
        #img=cv2.imread('C:\\archiveKaggle\\cars_train\\cars_train' + '\\'+str(NameImg)) 
       
        #img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        ImagePath=DataPath+ str(NameImg)
        # hay imagenes que vienen en blanco y negro
        img=cv2.imread(ImagePath)
        #print(img.shape)
        if int(img.shape[2]) !=3: continue
        TabImagePath.append(ImagePath)
        Modelo=int(lineadelTrain[5])
        Brand, BrandName=GetBrandFromModel(Modelo)
        if Brand==-1 :
            print ("NO SE ENCUENTRA MODELO " + str(Modelo) + " en " + NameImg)
        #if Brand >20: continue
        Y.append(Brand)
        #images.append(img)
        imagesName.append(NameImg)
       
    return TabImagePath, Y, imagesName


def process_image(image):
    
    # Process a PIL image for use in a PyTorch model
  
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}')

    """

    # Building image transform
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    """
    transform = transforms.Compose([transforms.Resize((244,244)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def predict(image_path, model, topk=5):
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted


    
###########################################################
# MAIN
##########################################################

from tensorflow.keras.models import load_model


TabImagePath_test, Y_test, imageName_test=loadimagesTest()

TotalHits=0
TotalFailures=0
with open( "BrandsResults.txt" ,"w") as  w:
    
    
    
    for i in range(len(TabImagePath_test)):
        
                
        TabP=[]
        TabModel=[]
        TabPredictions1=[]
        

        conf, predicted1=predict(TabImagePath_test[i], model_path, topk=5)
        #print(conf)
        #print(predicted1)
        #print(classes[predicted1[0]])
        IndexCarBrandPredict=int(classes[predicted1[0]])
        if (IndexCarBrandPredict > 48):
            print ("ERROR indice calculado " + str(IndexCarBrandPredict)+ " imagen= " +str(IndexCarBrandPredict))
            continue
        
        
        IndexCarBrandTrue=Y_test[i]
        NameCarBrandPredict=TabCarBrand[IndexCarBrandPredict]
        NameCarBrandTrue=TabCarBrand[IndexCarBrandTrue]
        if Y_test[i]!=IndexCarBrandPredict:
            TotalFailures=TotalFailures + 1
            print("ERROR " + imageName_test[i]+ " is assigned brand " + str(IndexCarBrandPredict)
                  +  NameCarBrandPredict + "  True brand is " + str(Y_test[i])+ NameCarBrandTrue)
                  
        else:
            print("HIT " + imageName_test[i]+ " is assigned brand " + str(IndexCarBrandPredict)
                  +  NameCarBrandPredict )
          
                 
          
            TotalHits=TotalHits+1
        lineaw=[]
        lineaw.append(imageName_test[i]) 
        lineaw.append(str(Y_test[i]))
        lineaw.append(NameCarBrandTrue)
        
        lineaw.append(NameCarBrandPredict)
        #lineaw.append( str(p))
        lineaWrite =','.join(lineaw)
        lineaWrite=lineaWrite + "\n"
        w.write(lineaWrite)
          
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%") 
