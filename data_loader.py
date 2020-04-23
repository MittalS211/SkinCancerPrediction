import os 
import random
from PIL import Image
from matplotlib import image
import numpy as np

def data_load():
    train_samples = []
    test_samples = []

    dirpath = "data/train/benign"

    files = os.listdir(dirpath)
    for file in files: 
        img = image.imread(dirpath+'/'+file)
        train_samples.append((img,0))

    dirpath = "data/train/malignant"

    files = os.listdir(dirpath)
    for file in files: 
        img = image.imread(dirpath+'/'+file)
        train_samples.append((img,1))  

    dirpath = "data/test/malignant"

    files = os.listdir(dirpath)
    for file in files: 
        img = image.imread(dirpath+'/'+file)
        test_samples.append((img,1))  

    dirpath = "data/test/benign"

    files = os.listdir(dirpath)
    for file in files: 
        img = image.imread(dirpath+'/'+file)
        test_samples.append((img,0))  
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    return np.asarray(train_samples),np.asarray(test_samples)


            