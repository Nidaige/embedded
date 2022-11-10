import os
import random
import math
import shutil

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
def getFirst(e):
    return e[0]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transformers):
        self.root = root
        self.transforms = transformers
        # load all image files, sorting them to
        # ensure that they are aligned
        self.classes = ['bar', 'bedroom_sleeping', 'classroom', 'gym', 'hall_downstairs', 'hall_upstairs', 'kitchen',
                   'library', 'living_room'
            , 'street_running', 'street_walking']
        self.imgs = []  # holds tuples of image paths and the index of the label
        categories = os.listdir("data/images/"+root)
        for category in categories:
            images = os.listdir("data/images/"+root+"/"+category)
            for image in images:
                self.imgs.append(["data/images/"+root+"/"+category+"/"+image, self.classes.index(category)])



    def __getitem__(self, idx):
        # load images and masks
        ImgPath,ImgClass = self.imgs[idx]
        img = self.transforms(Image.open(ImgPath).convert('RGB'))
        return(img,ImgClass)



    def __len__(self):
        return len(self.imgs)


def imshow(img,s=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if s and not '\n' in s:
        s = ' '.join(s.split())
        p = s.find(' ',int(len(s)/2))
        s = s[:p]+"\n"+s[p+1:]
    plt.text(0,-20,s)
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,(5,5))
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.fc1 = nn.Linear(16*29*29,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,11)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*29*29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():

    transform = transforms.Compose(
        [transforms.Resize((128,128)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CustomDataset("training",transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = CustomDataset("validation",transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)


    classes = [ 'bar','bedroom_sleeping','classroom','gym','hall_downstairs','hall_upstairs','kitchen','library','living_room'
        ,'street_running','street_walking']


    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    s = ' '.join('%5s' % classes[labels[j]] for j in range(16))
    print(s)
    imshow(torchvision.utils.make_grid(images),s)
    print(1)

    net = Net()
    #import torchvision.models as models
    #net = models.resnet18(pretrained=True)
    #net.fc = nn.Linear(512,10)
    #import pdb;pdb.set_trace()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(50):
        running_loss = 0.0
        j = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            j = i
            if(i%500==0):
                print(epoch,i,running_loss/(i+1))
        print("max i:",j)
    dataiter2 = iter(testloader)
    images2, labels2 = next(dataiter2)
    outputs = net(images2)
    _, predicted = torch.max(outputs,1)
    s1 = "Pred:"+' '.join('%5s' % classes[predicted[j]] for j in range(16))
    s2 = "Actual:"+' '.join('%5s' % classes[labels2[j]] for j in range(16))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels2[j]] for j in range(16)))
    imshow(torchvision.utils.make_grid(images2),s1+"\n"+s2)
    score = 0
    for a in range(len(predicted)):
        if predicted[a]==labels2[a]:
            score+=1
    print("Accuracy = "+str(score/16))

# Read text file with data from path, create image file.
def convertSampleToImage(path, max_values):
    data = []
    with open(path) as file:
        linecount = 0
        for line in file:
            try:
                num = float(line)+128  # parse read line as a float
                # Normalize to 0-255 by adding 128, and then cutting off anything outside 0-255
                if num < 0:
                    num = 0
                if num > 255:
                    num = 255
                if linecount < max_values or max_values == -1:  # if we still want more data / not full yet:
                    data.append(num)
                else:
                    break  #data is full, stop reading file.
            except:
                continue
            linecount += 1
    # Data is now list of lists of numbers 0-255.
    # Time to arrange in a grid:
    while len(data) < 600*600:
        data.append(0)
    data_array = np.array(data)

    grid = np.reshape(data_array, (600, 600))
    im = Image.fromarray(np.uint8(grid))
    return im

def iterate_dataFolder():
    counter = 0
    for category in ['bar','bedroom_sleeping','classroom','gym','hall_downstairs','hall_upstairs','kitchen','library','living_room','street_running','street_walking']:
        if category not in os.listdir("data/images"):
            os.mkdir("data/images/"+category)
        samples = os.listdir("data/"+category+"/")
        for sample in samples:
            path = 'data/'+category+'/'+sample+'/sound.txt'
            image = convertSampleToImage(path, 600*600)
            image.save("data/images/"+category+"/"+sample+".png")
            counter += 1
            if counter%1000==0:
                print("Processed",counter,"files")

def move_images_into_training_and_validation(split):
    # Create training and validation directories if they do not exist.
    if "training" not in os.listdir("data/images"):
        os.mkdir("data/images/" + "training")
    if "validation" not in os.listdir("data/images"):
        os.mkdir("data/images/" + "validation")

    # iterate through list of categories, see if they exist within training and validation directories, create if not
    for category in os.listdir("data/images"):
        if category not in os.listdir("data/images/training") and category not in ["training","validation"]:
            os.mkdir("data/images/training/"+category)
        if category not in os.listdir("data/images/validation") and category not in ["training","validation"]:
            os.mkdir("data/images/validation/"+category)

        # get name of all images for current category
        images = os.listdir("data/images/" + category)
        for image in images:
            if random.random() < split:  # divide into training/validation at a rate of <split>
                os.rename("data/images/" + category + "/" + image, "data/images/training/" + category + "/" + image)
            else:
                os.rename("data/images/" + category + "/" + image, "data/images/validation/" + category + "/" + image)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    #iterate_dataFolder()  # used to convert text files to images
    #move_images_into_training_and_validation(0.7)  # used to move files from category-based folders into dataset-based folders
    run()
    train()
