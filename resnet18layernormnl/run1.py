from resnet18_layernormnl import ResNet18
import os
import torch
import numpy as np
import codecs
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import sys

if len(sys.argv)==1:
	BATCH_SIZE = 1
else:
	BATCH_SIZE = int(sys.argv[1])

#BATCH_SIZE = 8
LR = 0.001
preEPOCH = 
EPOCH = 
NUM_WORKERS=2
CHECKPOINT_FILENAME="checkpoint.tar"
ORDER=1

TRAIN_DIR="/uufs/chpc.utah.edu/common/home/u1498392/wsi0.3125_exp/dev/0.3125dataset/train"
TRAIN_LABELFILE_PATH="/uufs/chpc.utah.edu/common/home/u1498392/wsi0.3125_exp/dev/0.3125dataset/trainlabel"
TEST_DIR="/uufs/chpc.utah.edu/common/home/u1498392/wsi0.3125_exp/dev/0.3125dataset/test"
TEST_LABELFILE_PATH="/uufs/chpc.utah.edu/common/home/u1498392/wsi0.3125_exp/dev/0.3125dataset/testlabel"

class CustomDataset(Dataset):
	def __init__(self, data_dir, label_file, transform=None):
		self.data_list = []
		self.transform = transform
		reader = codecs.open(label_file, 'r', 'utf-8')
		lines = reader.readlines()
		for line in lines:
			filename, label = line.split('\t')
			filename = str(filename.strip())
			label = int(label.strip())
			image_file_path = os.path.join(data_dir, filename)
			image = Image.open(image_file_path)
			if image.mode != 'RGB':
				print(f"{image_file_path} is not RGB!")
				continue
			self.data_list.append((image_file_path, label))

	def __len__(self):
		return len(self.data_list)
	
	def __getitem__(self, index):
		image_file_path, label = self.data_list[index]
		image = Image.open(image_file_path)
		if self.transform:
			image = self.transform(image)
		return (image, label)

def save_checkpoint(state, filename="checkpoint.tar"):
	torch.save(state, filename)
	#print("checkpoint")

def load_checkpoint(checkpoint, model, optimizer):
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	#print("load checkpoint")


if torch.cuda.is_available():
	device = torch.device("cuda")
	print("GPU is available()")
else:
	device = torch.device("cpu")
	print("No GPU available, using CPU instead")

transform_train = transforms.Compose([
	transforms.CenterCrop((800, 1280)), # CenterCrop the image to (height, width) = (800, 1280)
	transforms.ToTensor(), # Convert the image to a PyTorch tensor
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
trainset = CustomDataset(TRAIN_DIR, TRAIN_LABELFILE_PATH, transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

transform_test = transforms.Compose([
	transforms.CenterCrop((800, 1280)), # CenterCrop the image to (height, width) = (800, 1280)
	transforms.ToTensor(), # Convert the image to a PyTorch tensor
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
testset = CustomDataset(TEST_DIR, TEST_LABELFILE_PATH, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

#net = models.resnet18(weights='DEFAULT').to(device)
net = ResNet18(input_shape=[800, 1280], num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

# load checkpoint
load_checkpoint(torch.load(CHECKPOINT_FILENAME), net, optimizer)
net.to(device)

outputfilename = "_".join(["resnet18layernormnl", str(BATCH_SIZE), str(ORDER), "output.csv"])
fw = open(outputfilename, "w")
fw.write(f'Epoch\tLoss\tTrain_accuracy\tTest_accuracy\n')

#train & test
for epoch in range(preEPOCH, EPOCH):
	#train
	net.train()
	sum_loss = 0
	train_correct = 0
	train_total = 0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()

		#forward & backward
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		#aggregate loss, #total_train, #train_correct
		sum_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		train_total += labels.size(0)
		train_correct += predicted.eq(labels.data).cpu().sum()
	
	print("Epoch: %d | Loss: %.3f | Acc: %.3f%%" % (epoch+1, sum_loss/(i+1), 100*train_correct/train_total))
	if (epoch+1)%10 == 0:
		checkpoint = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
		save_checkpoint(checkpoint)

	#test
	print("Testing...")
	net.eval()
	test_correct = 0
	test_total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			test_total += labels.size(0)
			test_correct += (predicted == labels).sum()
		
		print("Test\'s Acc: %.3f%%" % (100 * test_correct / test_total))
	
	fw.write("%d\t%.3f\t%.3f%%\t%.3f%%\n" % (epoch+1, sum_loss/(i+1), 100*train_correct/train_total, 100*test_correct/test_total))
	

fw.close()
print('Done')

