# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:52:27 2023

@author: Alfonso Blanco
"""

import numpy as np
import cv2
import time

def GetBrandFromModel(Model):
    f=open("CarBrand.csv","r")
    
    for linea in f:
        lineadelTrain =linea.split(",")
        ModelFrom=int(lineadelTrain[1])
        ModelTo=int(lineadelTrain[2])
        if Model >= ModelFrom and Model <= ModelTo:
            return int(lineadelTrain[0]), lineadelTrain[3]
    print("RARO NO ENCUENTRA EL MODELO")
    return -1, ""
      
inicio=time.time()

dir="KaggleCarsByBrands_1_49"
f=open("cardatasettrain.csv","r")
ContTraining=0
ContValid=0
ContTest=0
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta==1: continue
    
    lineadelTrain =linea.split(",")
    NameImg=lineadelTrain[6]
    # OJO LLEVA UN CR AL FINAL
    NameImg=NameImg[0:9]
    #print(NameImg)
    Model =int(lineadelTrain[5])
    Brand, BrandName = GetBrandFromModel(Model)
    if Brand == -1:
        print ("Raro No se puede asignar marca a la imagen " + NameImg + " modelo "+ str(Model))
        continue
    #if Brand >20: continue
    print ("Se asigna la marca "+  str(BrandName)  +   " a la imagen " + NameImg + " modelo "+ str( Model))
 
    StrBrand=str(Brand)
    if len(StrBrand) < 2 : StrBrand="0"+StrBrand
    
    img=cv2.imread('C:\\archiveKaggle\\cars_train\\cars_train' + '\\'+ NameImg) 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
   
    if Conta > 8000:
        cv2.imwrite(dir +"\\test\\"+ NameImg,img)
        ContTest=ContTest+1
    else:
        if Conta > 7000:
            cv2.imwrite(dir +"\\valid\\"+ StrBrand +"\\"+NameImg,img)
            ContValid=ContValid+1
            
        else: 
            Sitio=dir +"\\train\\"+ StrBrand+"\\"+NameImg
            
            cv2.imwrite(Sitio,img)
            ContTraining=ContTraining+1
        
print("")         
print("Records to train = "+str(ContTraining))
print("Records to valid = "+str(ContValid))
print("Records to test = "+str(ContTest))
print("time in seconds = " + str(time.time()- inicio))
