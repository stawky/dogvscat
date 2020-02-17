import os
import random
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms

class DogvsCat(data.Dataset):
    def __init__(self,root,transform=None,train=True,test=False):
        self.test=test
        self.train=train
        self.transform=transform
        imgs=[os.path.join(root,img)for img in os .listdir(root)]

        if self.test:
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2]))

        imgs_num=len(imgs)
        
        if self.test:
            self.imgs=imgs
        else:
            random.shuffle(imgs)
            if self.train:
                self.imgs=imgs[:int(0.8*imgs_num)]
            else:
                self.imgs=imgs[int(0.8*imgs_num):]

    def __getitem__(self,index):
        img_path=self.imgs[index]
        if self.test:
            label=int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            if 'cat' in img_path.split('/')[-1]:
                label = 0
            else:
                label = 1
        data=Image.open(img_path)
        data=self.transform(data)
        return data,label
    
    def __len__(self):
        return len(self.imgs)

