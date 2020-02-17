import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np
import os
import config
from tensorboardX import SummaryWriter

from getdata import DogvsCat 
from network import resnet101

writer = SummaryWriter('saved/log_dir')
#load config
configs = config.Config()
configs_dict = configs.get_config()
num_workers =  configs_dict['num_workers']
batchSize = configs_dict['batchSize']
nepoch = configs_dict['nepoch']
lr = configs_dict['lr']
cuda = configs_dict['cuda']
optimizer = configs_dict['optimizer']
transform_train=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])


transform_val=transforms.Compose([ 
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

trainset=DogvsCat('./data/train',transform=transform_train)
valset  =DogvsCat('./data/train',transform=transform_val)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=batchSize,shuffle=True,num_workers=num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=batchSize,shuffle=False,num_workers=num_workers)

model=resnet101	(pretrained=True)
model.fc=nn.Linear(2048,2)
model.cuda()
if optimizer == 'Adam':
	optimizer=torch.optim.Adam(model.parameters(),lr=lr,
	betas=(0.9,0.999),
	eps=1e-08,weight_decay=0,amsgrad=False)
	scheduler = None
else:
	optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
	scheduler=StepLR(optimizer,step_size=2)
    

criterion=nn.CrossEntropyLoss()
criterion.cuda()
def train(epoch):
	print('\nEpoch: %d' % epoch)
	if scheduler:
		scheduler.step()
	model.train()
	for batch_idx,(img,label) in enumerate(trainloader):
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		
		print("Epoch:%d [%d|%d] loss:%f" %(epoch,batch_idx,len(trainloader),loss.mean()))
		writer.add_scalar('lr',lr,epoch)
		writer.add_scalar('loss',loss.mean(),epoch)


		
        
def val(epoch):
	bestacc = 0
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	acc = (1.0*correct.numpy())/total
	writer.add_scalar('acc',acc,epoch)
	if acc > bestacc:
		bestacc = acc
		torch.save(model.state_dict(),'saved/model/model.pth')


	
    
for epoch in range(nepoch):
	train(epoch)
	val(epoch)
