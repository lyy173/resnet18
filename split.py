import os
import random
import shutil

trainpath = r'C:\Users\liang\code\resnet\dataset\data\train'
valpath = r'C:\Users\liang\code\resnet\dataset\data\val'
trainlist = os.listdir(trainpath)
random.shuffle(trainlist)
train_n = len(trainlist)

val_n = int(train_n*0.2)
val_name = trainlist[:val_n]

for val in val_name:
    shutil.move(os.path.join(trainpath,val),os.path.join(valpath,val))