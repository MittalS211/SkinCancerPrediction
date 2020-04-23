import numpy as np 
from data_loader import data_load
from model import Model

train, test =  data_load()

m = train.shape[0]
pix = train[1][0].shape[1]

train_x = []
train_y = []
test_x = []
test_y = []
for i in train:
    train_x.append(i[0])
    train_y.append(i[1])
for i in test:
    test_x.append(i[0])
    test_y.append(i[1])    

train_x = np.asarray(train_x)    
test_x = np.asarray(test_x)  
train_y = np.asarray(train_y)
test_y = np.asarray(test_y)

train_x = train_x.reshape(train_x.shape[0],-1).T
test_x = test_x.reshape(test_x.shape[0],-1).T


train_x = train_x/255
test_x = test_x/255

d = Model(train_x,train_y,test_x,test_y,2000,0.5)
