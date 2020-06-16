"""Face classification project using PyTorch and custom data set"""





# importing required libraries

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as dl
import torch.optim as optim
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy

#Show a batch of images
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()



# transforming image
composed = transforms.Compose([transforms.Resize((250,250)), transforms.ToTensor(), transforms.Normalize([0.4,0.4,0.4],[0.4,0.4,0.4])])

# loading data Training/Test
train_dataset = dsets.ImageFolder(root="./images/training", transform=composed)
test_dataset = dsets.ImageFolder(root="./images/test", transform=composed)
class_names = train_dataset.classes
print("labels:", class_names)

# data loading
train_loader = dl(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = dl(dataset=test_dataset, batch_size=2, shuffle=True)


# Convolution neural network class

class CNN(nn.Module):

    #Constructor
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels= 16, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16*62*62, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 40)
        self.bn_fc2 = nn.BatchNorm1d(40)


    # prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16*62*62)
        x = torch.dropout(x,p=0.5, train=True)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        return x

# creating model using CNN Class
model = CNN()

# defining hyper-parameters

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), learning_rate)


# training of model

epochs = 10
cost_list = list = []
accuracy_list = []
N_test = len(test_dataset)

def train_model(epochs):
    for epoch in range(epochs):
        cost = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()
        cost_list.append(cost)
        print("Epoch",epoch+1,"loss",cost/100)



train_model(epochs)

my_model = CNN()
PATH = "./faceclass.pth"
torch.save(my_model.state_dict(), PATH)


dataiter = iter(test_loader)
testimages, testlabels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(testimages))
print('GroundTruth: ', [class_names[x] for x in testlabels])


my_model.load_state_dict((torch.load(PATH)))

output = my_model(testimages)
predicted = torch.max(output, 1)

print('Predicted: ', [class_names[x] for x in testlabels])
