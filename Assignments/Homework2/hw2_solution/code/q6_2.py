import torch
import numpy as np
from math import sqrt
import torch.nn as nn



def loader(dictionary):

    print("Loading Train data")
    X_train = []
    y_train = []


    f = open('data/train.txt')
    for l in f:
        y_train.append(int(l[0]))
        line = l[2:].split()
        temp = []
        for item in line:
            if item in dictionary:
                temp.append(dictionary[item])

        X_train.append(torch.tensor(temp))


    y_train = np.asarray(y_train).reshape(-1,1)



    print("Loading Test data")
    X_test = []
    y_test = []

    f = open('data/test.txt')
    for l in f:
        y_test.append(int(l[0]))
        line = l[2:].split()
        temp = []
        for item in line:
            if item in dictionary:
                temp.append(dictionary[item])

        X_test.append(torch.tensor(temp))


    y_test = np.asarray(y_test).reshape(-1,1)



    print("Loading Unlabelled data")
    X_unlabelled = []


    f = open('data/unlabelled.txt')
    for l in f:
        line = l[2:].split()
        temp = []
        for item in line:
            if item in dictionary:
                temp.append(dictionary[item])

        X_unlabelled.append(torch.tensor(temp))




    return X_train, y_train, X_test, y_test, X_unlabelled


class q6_2(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.embed = nn.Embedding(dim, 20)
        self.fc = nn.Linear(20, 1)


    def init_weights(self):
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        z = [self.embed(item) for item in x]
        avg = [torch.mean(item, dim=0) for item in z]
        z = self.fc(torch.stack(avg))
        h = torch.sigmoid(z)
        return h


def train(X_train,y_train, net, criterion, optimizer):
    for epoch in range(8):  # loop over the dataset multiple times

        optimizer.zero_grad()
        loss = criterion(net(X_train).float(), y_train.float())
        loss.backward()
        optimizer.step()

        print('loss: %.3f ' %loss.item())


    print('Finished Training')




def test(X_test, y_test, net):
    correct = 0
    total = 0
    with torch.no_grad():
        output = net(X_test)
        for i in range(len(output)):
            if output[i] > 0.5 and y_test[i] == 1:
                correct += 1
            elif output[i] <= 0.5 and y_test[i] == 0:
                correct += 1
        total = len(y_test)
    print('Accuracy: %d %%' % (
        100 * correct / total))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dictionary = {}
    index = 0
    with open('data/train.txt', 'r') as f:
        for l in f:
            line = l[2:].split()
            for item in line:
                if item not in dictionary:
                    dictionary[item] = index
                    index += 1


    X_train, y_train, X_test, y_test, X_unlabelled = loader(dictionary)

    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)



    net = q6_2(len(dictionary)).to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.5)

    train(X_train, y_train, net, criterion, optimizer)
    test(X_test, y_test, net)

    f = open('output/predictions_q2.txt', 'w')

    printout = []
    with torch.no_grad():
        output = net(X_test)
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

