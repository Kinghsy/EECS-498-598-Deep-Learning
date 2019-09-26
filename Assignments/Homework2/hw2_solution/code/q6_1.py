import torch
import numpy as np
from math import sqrt
import torch.nn as nn
import torch.utils.data as data_utils
import time
from sklearn.feature_extraction.text import CountVectorizer


def loader():

    print("Loading Train data")
    y_train = []
    corpus1 = []

    f = open('data/train.txt')
    for l in f:
        y_train.append(int(l[0]))
        corpus1.append(l[2:])

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(corpus1).toarray()

    X_train[X_train > 1] = 1
    y_train = np.asarray(y_train).reshape(-1,1)



    print("Loading Test data")
    y_test = []
    corpus2 = []

    f = open('data/test.txt')
    for l in f:
        y_test.append(int(l[0]))
        corpus2.append(l[2:])


    X_test = vectorizer.transform(corpus2).toarray()

    X_test[X_test > 1] = 1
    y_test = np.asarray(y_test).reshape(-1,1)



    print("Loading Unlabelled data")
    corpus3 = []

    f = open('data/unlabelled.txt')
    for l in f:
        corpus3.append(l[2:])


    X_unlabelled = vectorizer.transform(corpus3).toarray()

    X_unlabelled[X_unlabelled > 1] = 1

    return X_train, y_train, X_test, y_test, X_unlabelled


class q6_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(7421, 1)


    def init_weights(self):
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        z = self.fc(x)
        h = torch.sigmoid(z)
        return h


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(8):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (representations, labels) in enumerate(trainloader):
            representations = representations.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            output = net(representations)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')




def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            representations, labels = data
            representations = representations.to(device).float()
            labels = labels.to(device).float()
            outputs = net(representations)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    print('Accuracy: %d %%' % (
        100 * correct / total))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test, X_unlabelled = loader()

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    X_unlabelled = torch.tensor(X_unlabelled)

    trainset = data_utils.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = data_utils.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)

    unlabelledset = data_utils.TensorDataset(X_unlabelled)
    unlabelledloader = data_utils.DataLoader(unlabelledset, batch_size=100, shuffle=False)

    net = q6_1().to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)

    f = open('output/predictions_q1.txt', 'w')

    for data in unlabelledloader:
        info, = data
        output = net(info.to(device).float())
        output[output < 0.5] = int(0)
        output[output >= 0.5] = int(1)
        for item in output:
            f.write(str(int(item.item())))
            f.write("\n")
    f.close()

if __name__ == "__main__":
    main()

