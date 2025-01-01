from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,datapath,classdict,transforms=None):
        self.datalist = os.listdir(datapath)
        self.transforms = transforms
        self.classdict = classdict
        self.datapath = datapath
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.datapath,self.datalist[idx]))
        if self.transforms:
            img = self.transforms(img)
        label = self.classdict[self.datalist[idx].split('.')[0]]
        return img,label