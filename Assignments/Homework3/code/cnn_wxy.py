import torch
import numpy as np
from math import sqrt
import torch.nn as nn
import time


def loader(dictionary):

    print("Loading Train data")
    X_train = []
    y_train = []


    f = open('data/train.txt')
    for l in f:
        y_train.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            else:
                temp.append(len(dictionary))
                count +=1
            if count == 15:
                break

        while count < 15:
            temp.append(len(dictionary)+1)
            count += 1
            if count == 15:
                break

        X_train.append(temp)


    y_train = np.asarray(y_train).reshape(-1,1)



    print("Loading Test data")
    X_test = []
    y_test = []

    f = open('data/test.txt')
    for l in f:
        y_test.append(int(l[0]))
        line = l[2:].split()
        temp = []
        count = 0

        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            else:
                temp.append(len(dictionary))
                count +=1
            if count == 15:
                break

        while count < 15:
            temp.append(len(dictionary) + 1)
            count += 1
            if count == 15:
                break

        X_test.append(temp)


    y_test = np.asarray(y_test).reshape(-1,1)



    print("Loading Unlabelled data")
    X_unlabelled = []


    f = open('data/unlabelled.txt')
    for l in f:
        line = l[2:].split()
        temp = []
        count = 0
        for item in line:

            if item in dictionary:
                temp.append(dictionary[item])
                count += 1
            else:
                temp.append(len(dictionary))
                count +=1

            if count == 15:
                break

        while count < 15:
            temp.append(len(dictionary) + 1)
            count += 1
            if count == 15:
                break

        X_unlabelled.append(temp)


    return X_train, y_train, X_test, y_test, X_unlabelled

class CNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.F = 5     #  this is the number of feature maps
        self.mode = 'max'
        self.embed = nn.Embedding(dim, 300)
        self.Conv = nn.Conv1d(300,128,kernel_size = 7)
        if self.mode == 'avg':
            self.pool = nn.AvgPool1d(15-self.F+1)
        elif self.mode =='max':
            #self.pool = nn.MaxPool1d(15 - self.F + 1)
            self.pool = nn.MaxPool1d(15 - 7 + 1)
        self.nolinear = nn.ReLU()
        self.fc = nn.Linear(128, 1)

    def init_weights(self):
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):

        print("x.shape is ", x.size())
        x1 = self.embed(x)
        print("x1.shape is ", x1.size())
        x1 = x1.permute(0,2,1)
        print("x1.shape is ", x1.size())
        x2 = self.Conv(x1)
        print("x2.shape is ", x2.size())
        x3 = self.pool(x2)
        print("x3.shape is ", x3.size())
        x3 = self.nolinear(x3)
        x3 = x3.view(-1, 128)
        print("x3.shape is ", x3.size())
        x4 = self.fc(x3)
        print("x4.shape is ", x4.size())
        z = torch.sigmoid(x4)
        return z


def train(X_train,y_train, net, criterion, optimizer):

    Xtrain = X_train
    labels = y_train
    epoch =20
    batchsize=1000
    for epochi in range(epoch):

        start = time.time()

        batchnum = len(Xtrain) // batchsize

        for i in range(batchnum):

            batch_idx = np.random.choice(len(Xtrain), batchsize, replace=False)
            Xtrain_batch = Xtrain[batch_idx, :]
            labels_batch = labels[batch_idx]

            optimizer.zero_grad()

            scores = net(Xtrain_batch)
            running_loss = criterion(scores, labels_batch.float())

            running_loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            if i % 10 == 0 or i == batchnum - 1:
                end = time.time()
                print(
                    '[epoch %d, iter %3d] \n     - loss: %.8f       - eplased time %.3f' %
                    (epochi + 1, i + 1, running_loss.item() / batchsize,  end - start))
                start = time.time()
    print('Finished Training')


def test(X_test, y_test, net):
    correct = 0
    total = 0
    ytest = y_test
    with torch.no_grad():
        output = net(X_test)
        for i in range(len(output)):
            if output[i] > 0.5 and ytest[i] == 1:
                correct += 1
            elif output[i] <= 0.5 and ytest[i] == 0:
                correct += 1
        total = len(ytest)
    print('Accuracy: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    dictionary = {}
    index = 0
    u=[]
    with open('data/train.txt', 'r') as f:
        for l in f:
            line = l[2:].split()
            u.append(len(line))
            for item in line:
                if item not in dictionary:
                    dictionary[item] = index
                    index += 1
    X_train, y_train, X_test, y_test, X_unlabelled = loader(dictionary)

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    X_unlabelled = torch.tensor(X_unlabelled).to(device)



    net = CNN(len(dictionary)+2).to(device)
    net.init_weights()
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    train(X_train, y_train, net, criterion, optimizer)
    test(X_test, y_test, net)

    f = open('output/predictions_HW3_q1.txt', 'w+')

    printout = []
    with torch.no_grad():
        output = net(X_test.to(device))
        for i in range(len(output)):
            if output[i] > 0.5:
                printout.append(1)
            elif output[i] <= 0.5:
                printout.append(0)



    for item in printout:
        f.write(str(int(item)))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()
