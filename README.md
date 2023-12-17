# CarsBrands_Resnet_Pytorch
Project that detects the brand of a car, between 1 and 49 brands ( the 49 brands of Stanford car file), that appears in a photograph with a success rate of more than 80% (using a test file that has not been involved in the training as a valid or training file, "unseen data") and can be implemented on a personal computer.

The project is an adaptation and simplification of the one presented at https://www.kaggle.com/code/hussnain47/car-object-detection-and-classification to consider the detection only of car brands instead of detecting brand and model.

All used packages, if any are missing, can be installed with a simple pip after de error of missing.

For this, the Stanford car file downloaded from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/code?resource=download has been used.

When you download and unzip the download file you will see that it contains the directories
archive\cars_train\cars_train with 8144 numbered images. This folder should be copied to the c: directory so that it is accessible by other projects apart from CarsBrands_Resnet_Pytorch and also change the name of the archive folder to archiveKaggle so that there is a folder with C:\\archiveKaggle\\cars_train\ \cars_train

Download all the files that accompany the CarsBrands_Resnet_Pytorch project in a single folder.

The director file is cardatasettrain.csv downloaded from:
https://github.com/BotechEngineering/StanfordCarsDatasetCSV/tree/main


For the training, images 1 to 7000 will be considered as train/valid making an internal split of 70% train and 30% valid, and from 8000 to 8144 as an independent test of the training process.

You have to create the folder structure that Resnet requires: a directory with 3 folders: train, valid and test. From train and valid hang as many folders as car brands are considered named with the code assigned to each brand. This structure is created by running the program in the download folder of the project:

CreateDirTrainTestCarBrands_1_49.py

Next, the structure created in the previous step is filled in by executing:

FillDirKaggleCarsByBrand_1_49.py

As part of the downloaded Stanford file is supposed to be located in C:\\archiveKaggle\\cars_train\\cars_train, if not, modify the FillDirKaggleCarsByBrand_1_49.py line 52 so that it points to where the Stanford file is located.

Run the train:

TrainCarsBrands_Resnet_Pytorch.py

which comes ready for 20 epoch, designed so that it can run in a reasonable time on a laptop.
really 10 epoch would be enough

Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to C:\Users\Alfonso Blanco/.cache\torch\hub\checkpoints\resnet50-0676ba61.pth

  0%|                                                                                  | 0.00/97.8M [00:00<?, ?B/s]
  2%|█▋                                                                       | 2.21M/97.8M [00:00<00:04, 21.2MB/s]
  5%|███▎                                                                     | 4.43M/97.8M [00:00<00:04, 21.2MB/s]
  7%|█████▏                                                                   | 6.90M/97.8M [00:00<00:04, 22.3MB/s]
  9%|██████▋                                                                  | 9.03M/97.8M [00:00<00:04, 21.6MB/s]
 11%|████████▎                                                                | 11.1M/97.8M [00:00<00:04, 19.8MB/s]
 13%|█████████▋                                                               | 13.0M/97.8M [00:00<00:04, 19.3MB/s]
 15%|███████████                                                              | 14.9M/97.8M [00:00<00:04, 18.0MB/s]
 18%|█████████████▎                                                           | 17.9M/97.8M [00:00<00:03, 21.4MB/s]
 21%|███████████████▍                                                         | 20.6M/97.8M [00:01<00:03, 22.8MB/s]
 23%|█████████████████                                                        | 22.8M/97.8M [00:01<00:03, 22.4MB/s]
 26%|██████████████████▋                                                      | 25.0M/97.8M [00:01<00:03, 21.8MB/s]
 29%|█████████████████████▍                                                   | 28.7M/97.8M [00:01<00:02, 26.0MB/s]
 32%|███████████████████████▍                                                 | 31.5M/97.8M [00:01<00:02, 26.1MB/s]
 35%|█████████████████████████▎                                               | 34.0M/97.8M [00:01<00:02, 25.4MB/s]
 38%|███████████████████████████▌                                             | 36.9M/97.8M [00:01<00:02, 26.1MB/s]
 40%|█████████████████████████████▍                                           | 39.4M/97.8M [00:01<00:02, 25.5MB/s]
 43%|███████████████████████████████▋                                         | 42.4M/97.8M [00:01<00:02, 26.5MB/s]
 46%|█████████████████████████████████▌                                       | 44.9M/97.8M [00:02<00:02, 25.8MB/s]
 48%|███████████████████████████████████▎                                     | 47.4M/97.8M [00:02<00:02, 25.2MB/s]
 51%|█████████████████████████████████████▏                                   | 49.8M/97.8M [00:02<00:02, 22.6MB/s]
 54%|███████████████████████████████████████▏                                 | 52.4M/97.8M [00:02<00:02, 23.3MB/s]
 56%|████████████████████████████████████████▊                                | 54.7M/97.8M [00:02<00:02, 21.0MB/s]
 59%|██████████████████████████████████████████▉                              | 57.5M/97.8M [00:02<00:01, 22.8MB/s]
 61%|████████████████████████████████████████████▋                            | 59.9M/97.8M [00:02<00:01, 22.8MB/s]
 64%|██████████████████████████████████████████████▎                          | 62.1M/97.8M [00:02<00:01, 19.9MB/s]
 66%|████████████████████████████████████████████████▌                        | 65.0M/97.8M [00:02<00:01, 21.9MB/s]
 70%|██████████████████████████████████████████████████▉                      | 68.2M/97.8M [00:03<00:01, 24.5MB/s]
 74%|█████████████████████████████████████████████████████▉                   | 72.3M/97.8M [00:03<00:00, 28.6MB/s]
 77%|████████████████████████████████████████████████████████                 | 75.1M/97.8M [00:03<00:00, 24.1MB/s]
 79%|█████████████████████████████████████████████████████████▉               | 77.6M/97.8M [00:03<00:00, 23.9MB/s]
 82%|███████████████████████████████████████████████████████████▋             | 79.9M/97.8M [00:03<00:00, 23.6MB/s]
 84%|█████████████████████████████████████████████████████████████▍           | 82.3M/97.8M [00:03<00:00, 21.5MB/s]
 86%|███████████████████████████████████████████████████████████████          | 84.4M/97.8M [00:03<00:00, 21.1MB/s]
 88%|████████████████████████████████████████████████████████████████▌        | 86.5M/97.8M [00:03<00:00, 19.9MB/s]
 91%|██████████████████████████████████████████████████████████████████▍      | 89.0M/97.8M [00:04<00:00, 21.7MB/s]
 93%|████████████████████████████████████████████████████████████████████     | 91.1M/97.8M [00:04<00:00, 19.3MB/s]
 95%|█████████████████████████████████████████████████████████████████████▍   | 93.1M/97.8M [00:04<00:00, 16.4MB/s]
 97%|██████████████████████████████████████████████████████████████████████▋  | 94.7M/97.8M [00:04<00:00, 14.0MB/s]
 98%|███████████████████████████████████████████████████████████████████████▊ | 96.2M/97.8M [00:04<00:00, 11.2MB/s]
100%|████████████████████████████████████████████████████████████████████████▋| 97.4M/97.8M [00:05<00:00, 9.59MB/s]
100%|█████████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:05<00:00, 19.8MB/s]

No. epochs: 2,             Training Loss: 0.063             Valid Loss: 2.896             Valid Accuracy: 0.238

No. epochs: 3,             Training Loss: 0.078             Valid Loss: 1.894             Valid Accuracy: 0.488

No. epochs: 4,             Training Loss: 0.077             Valid Loss: 1.634             Valid Accuracy: 0.558

No. epochs: 5,             Training Loss: 0.065             Valid Loss: 1.317             Valid Accuracy: 0.625

No. epochs: 6,             Training Loss: 0.053             Valid Loss: 1.237             Valid Accuracy: 0.662

No. epochs: 7,             Training Loss: 0.046             Valid Loss: 1.337             Valid Accuracy: 0.64

No. epochs: 8,             Training Loss: 0.035             Valid Loss: 0.718             Valid Accuracy: 0.806

No. epochs: 9,             Training Loss: 0.032             Valid Loss: 0.678             Valid Accuracy: 0.816

No. epochs: 10,             Training Loss: 0.03             Valid Loss: 0.654             Valid Accuracy: 0.825

No. epochs: 11,             Training Loss: 0.031             Valid Loss: 0.641             Valid Accuracy: 0.823

No. epochs: 12,             Training Loss: 0.033             Valid Loss: 0.628             Valid Accuracy: 0.828

No. epochs: 13,             Training Loss: 0.037             Valid Loss: 0.638             Valid Accuracy: 0.83

No. epochs: 14,             Training Loss: 0.038             Valid Loss: 0.625             Valid Accuracy: 0.833

No. epochs: 15,             Training Loss: 0.039             Valid Loss: 0.637             Valid Accuracy: 0.827

No. epochs: 16,             Training Loss: 0.043             Valid Loss: 0.635             Valid Accuracy: 0.827

No. epochs: 17,             Training Loss: 0.047             Valid Loss: 0.645             Valid Accuracy: 0.826

No. epochs: 18,             Training Loss: 0.047             Valid Loss: 0.644             Valid Accuracy: 0.82

No. epochs: 19,             Training Loss: 0.053             Valid Loss: 0.63             Valid Accuracy: 0.826

No. epochs: 20,             Training Loss: 0.056             Valid Loss: 0.64             Valid Accuracy: 0.824

Test accuracy of model: 80.4%

>>> 

Test with unseen data

GuessCarsBrands_Resnet_Pytorch.py


HIT 08000.jpg is assigned brand 35Mercedes-Benz

HIT 08001.jpg is assigned brand 27Jeep

ERROR 08002.jpg is assigned brand 37Nissan
  True brand is 35Mercedes-Benz

HIT 08003.jpg is assigned brand 23Hyundai

HIT 08004.jpg is assigned brand 5BMW

HIT 08005.jpg is assigned brand 2Acura

HIT 08006.jpg is assigned brand 23Hyundai

ERROR 08007.jpg is assigned brand 3Astom Martin
  True brand is 5BMW

HIT 08008.jpg is assigned brand 10Chevrolet

HIT 08009.jpg is assigned brand 35Mercedes-Benz

HIT 08010.jpg is assigned brand 4Audi

HIT 08011.jpg is assigned brand 1AM

HIT 08012.jpg is assigned brand 28Lamborghini

ERROR 08013.jpg is assigned brand 10Chevrolet
  True brand is 19GMC

HIT 08014.jpg is assigned brand 4Audi

ERROR 08015.jpg is assigned brand 4Audi
  True brand is 6Bentley

HIT 08016.jpg is assigned brand 2Acura

HIT 08017.jpg is assigned brand 10Chevrolet

ERROR 08018.jpg is assigned brand 10Chevrolet
  True brand is 2Acura

HIT 08019.jpg is assigned brand 23Hyundai

HIT 08020.jpg is assigned brand 13Dodge

HIT 08021.jpg is assigned brand 5BMW

HIT 08022.jpg is assigned brand 28Lamborghini

HIT 08023.jpg is assigned brand 24Infiniti

HIT 08024.jpg is assigned brand 33Mazda

HIT 08025.jpg is assigned brand 19GMC

HIT 08026.jpg is assigned brand 4Audi

HIT 08027.jpg is assigned brand 4Audi

HIT 08028.jpg is assigned brand 4Audi

HIT 08029.jpg is assigned brand 18Ford

ERROR 08030.jpg is assigned brand 44Suzuki
  True brand is 15FIAT

HIT 08031.jpg is assigned brand 1AM

HIT 08032.jpg is assigned brand 5BMW

HIT 08033.jpg is assigned brand 5BMW

HIT 08034.jpg is assigned brand 23Hyundai

HIT 08035.jpg is assigned brand 10Chevrolet

ERROR 08036.jpg is assigned brand 23Hyundai
  True brand is 46Toyota

ERROR 08037.jpg is assigned brand 19GMC
  True brand is 10Chevrolet

HIT 08038.jpg is assigned brand 11Chrysler

HIT 08039.jpg is assigned brand 27Jeep

ERROR 08040.jpg is assigned brand 7Bugatti
  True brand is 10Chevrolet

HIT 08041.jpg is assigned brand 35Mercedes-Benz

HIT 08042.jpg is assigned brand 8Buick

HIT 08043.jpg is assigned brand 4Audi

HIT 08044.jpg is assigned brand 3Astom Martin

HIT 08045.jpg is assigned brand 23Hyundai

HIT 08046.jpg is assigned brand 46Toyota

HIT 08047.jpg is assigned brand 11Chrysler

ERROR 08048.jpg is assigned brand 26Jaguar
  True brand is 5BMW

HIT 08049.jpg is assigned brand 10Chevrolet

ERROR 08050.jpg is assigned brand 46Toyota
  True brand is 13Dodge

HIT 08051.jpg is assigned brand 3Astom Martin

HIT 08052.jpg is assigned brand 29Land Rover

HIT 08053.jpg is assigned brand 4Audi

HIT 08054.jpg is assigned brand 18Ford

HIT 08055.jpg is assigned brand 4Audi

HIT 08056.jpg is assigned brand 19GMC

ERROR 08057.jpg is assigned brand 19GMC
  True brand is 25Isuzu

HIT 08058.jpg is assigned brand 5BMW

HIT 08059.jpg is assigned brand 19GMC

ERROR 08060.jpg is assigned brand 11Chrysler
  True brand is 27Jeep

ERROR 08061.jpg is assigned brand 13Dodge
  True brand is 35Mercedes-Benz

HIT 08062.jpg is assigned brand 10Chevrolet

HIT 08063.jpg is assigned brand 19GMC

HIT 08064.jpg is assigned brand 23Hyundai

HIT 08065.jpg is assigned brand 27Jeep

ERROR 08066.jpg is assigned brand 23Hyundai
  True brand is 18Ford

ERROR 08067.jpg is assigned brand 4Audi
  True brand is 28Lamborghini

HIT 08068.jpg is assigned brand 5BMW

HIT 08069.jpg is assigned brand 29Land Rover

HIT 08070.jpg is assigned brand 10Chevrolet

HIT 08071.jpg is assigned brand 48Volvo

HIT 08072.jpg is assigned brand 22Honda

HIT 08073.jpg is assigned brand 6Bentley

HIT 08074.jpg is assigned brand 10Chevrolet

HIT 08075.jpg is assigned brand 10Chevrolet

ERROR 08076.jpg is assigned brand 2Acura
  True brand is 17FisKer

HIT 08077.jpg is assigned brand 10Chevrolet

HIT 08078.jpg is assigned brand 44Suzuki

HIT 08079.jpg is assigned brand 18Ford

HIT 08080.jpg is assigned brand 10Chevrolet

HIT 08081.jpg is assigned brand 4Audi

HIT 08082.jpg is assigned brand 5BMW

HIT 08083.jpg is assigned brand 35Mercedes-Benz

HIT 08084.jpg is assigned brand 23Hyundai

HIT 08085.jpg is assigned brand 3Astom Martin

ERROR 08086.jpg is assigned brand 5BMW
  True brand is 4Audi

HIT 08087.jpg is assigned brand 20Geo

HIT 08088.jpg is assigned brand 47Volkswagen

HIT 08089.jpg is assigned brand 22Honda

HIT 08090.jpg is assigned brand 13Dodge

HIT 08091.jpg is assigned brand 13Dodge

HIT 08092.jpg is assigned brand 13Dodge

HIT 08093.jpg is assigned brand 16Ferrari

HIT 08094.jpg is assigned brand 10Chevrolet

HIT 08095.jpg is assigned brand 7Bugatti

HIT 08096.jpg is assigned brand 3Astom Martin

HIT 08097.jpg is assigned brand 10Chevrolet

HIT 08098.jpg is assigned brand 44Suzuki

ERROR 08099.jpg is assigned brand 10Chevrolet
  True brand is 18Ford

HIT 08100.jpg is assigned brand 11Chrysler

HIT 08101.jpg is assigned brand 37Nissan

ERROR 08102.jpg is assigned brand 44Suzuki
  True brand is 35Mercedes-Benz

HIT 08103.jpg is assigned brand 46Toyota

HIT 08104.jpg is assigned brand 46Toyota

ERROR 08105.jpg is assigned brand 10Chevrolet
  True brand is 13Dodge

HIT 08106.jpg is assigned brand 28Lamborghini

HIT 08107.jpg is assigned brand 11Chrysler

HIT 08108.jpg is assigned brand 10Chevrolet

HIT 08109.jpg is assigned brand 29Land Rover

HIT 08110.jpg is assigned brand 46Toyota

HIT 08111.jpg is assigned brand 37Nissan

HIT 08112.jpg is assigned brand 13Dodge

HIT 08113.jpg is assigned brand 18Ford

ERROR 08114.jpg is assigned brand 13Dodge
  True brand is 35Mercedes-Benz

HIT 08115.jpg is assigned brand 34McLaren

HIT 08116.jpg is assigned brand 35Mercedes-Benz

HIT 08117.jpg is assigned brand 10Chevrolet

HIT 08118.jpg is assigned brand 5BMW

ERROR 08119.jpg is assigned brand 9Cadillac
  True brand is 10Chevrolet

HIT 08120.jpg is assigned brand 22Honda

ERROR 08121.jpg is assigned brand 9Cadillac
  True brand is 28Lamborghini

HIT 08122.jpg is assigned brand 9Cadillac

HIT 08123.jpg is assigned brand 42Scion

HIT 08124.jpg is assigned brand 6Bentley

HIT 08125.jpg is assigned brand 10Chevrolet

ERROR 08126.jpg is assigned brand 23Hyundai
  True brand is 18Ford

HIT 08127.jpg is assigned brand 2Acura

HIT 08128.jpg is assigned brand 18Ford

HIT 08129.jpg is assigned brand 7Bugatti

HIT 08130.jpg is assigned brand 7Bugatti

HIT 08131.jpg is assigned brand 19GMC

ERROR 08132.jpg is assigned brand 10Chevrolet
  True brand is 28Lamborghini

HIT 08133.jpg is assigned brand 2Acura

HIT 08134.jpg is assigned brand 21HUMMER


Total hits = 109
Total failures = 27
Accuracy = 80.1470588235294%
>>> 

Shows the results of the test with images from 8000.jpg to 8134.jpg, which have not been used as train or valid.



References:

https://www.kaggle.com/code/hussnain47/car-object-detection-and-classification

https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/code?resource=download

https://github.com/BotechEngineering/StanfordCarsDatasetCSV/blob/main/cardatasettest.csv

https://medium.com/analytics-vidhya/top-4-pre-trained-models-for-image-classification-with-python-code-a3cb5846248b

https://github.com/afaq-ahmad/Car-Models-and-Make-Classification-Standford_Car_dataset-mobilenetv2-imagenet-93-percent-accuracy/blob/master/Car_classification.ipynb

https://github.com/ablanco1950/CarsBrands_Inceptionv3
Comparing with this project, not only the success rate is improved (from 70% to 80%) but also the training time, going from needing 900 epochs to only 10.
