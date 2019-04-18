import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
np.random.seed(498)
import random
random.seed(498)
from PIL import Image
import torch
torch.manual_seed(498)
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(65, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "att_faces/training/"
    testing_dir = "att_faces/testing/"
    train_batch_size = 32
    train_number_epochs = 200

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True,should_gray=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.should_gray = should_gray        

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        if self.should_gray:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #TODO: design the architecture
        # Conv(in_channel=1, out_channel=8, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm
        # Conv(in_channel=8, out_channel=16, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm
        # Conv(in_channel=16, out_channel=32, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )


        self.fc = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
        )
        # Fully connected(out_features=1024)
        # ReLU
        # Fully connected(out_features=512)
        # ReLU
        # Fully connected(out_features=8)

    def forward_once(self, x):
        #TODO: implement the forward pass to get features for input image
        x1 = self.conv(x)
        x1s = x1.size()
        x2 = x1.view(x1s[0], -1)
        x3 = self.fc(x2)
        return x3

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #TODO: argument output1 is f(x1), output2 is f(x2)
        # calculate the contrasive loss and return it
        # Note that in this function we have batch of samples as input.
        euc_dist = F.pairwise_distance(output1, output2)
        loss_p1 = torch.pow(euc_dist, 2)
        loss_p2 = torch.pow(torch.clamp(self.margin - euc_dist, min=0.0), 2)
        loss_cont = torch.mean((1-label)*loss_p1 + label*loss_p2)
        return loss_cont


def evaluate(dataiter, net, split, device):
    for i in range(2):
        x0,_,_ = next(dataiter)
        for j in range(10):
            _,x1,_ = next(dataiter)
            concatenated = torch.cat((x0,x1),0)
            output1,output2 = net(Variable(x0).to(device),Variable(x1).to(device))
            euclidean_distance = F.pairwise_distance(output1, output2)
            imshow(torchvision.utils.make_grid(concatenated),'%s, dissimilarity:%.2f'%(split, euclidean_distance.item())) 
            plt.savefig('%s_%d_%d.png'%(split,i, j))
            plt.close()


if __name__ == "__main__":
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=10, batch_size=Config.train_batch_size)   
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=10, batch_size=1)
    dataiter = iter(train_dataloader)
    evaluate(dataiter, net, 'train', device)
 
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,num_workers=10,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)
    evaluate(dataiter, net, 'test', device)

    pickle.dump(loss_history, open('./loss_his.txt', 'wb'))
    plt.plot(loss_history)
    plt.show()
    plt.savefig("loss_his.png")

