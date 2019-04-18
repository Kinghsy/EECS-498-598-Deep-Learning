import torch
import numpy as np
import torch.nn as nn
import math
import time

class CNN(nn.Module):
    def __init__(self, dict_dim, hidden_dim, feature_dim, conv_k_size, pool_mode, out_dim):
        super().__init__()
        self.dict_dim = dict_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.conv_k_size = conv_k_size
        self.pool_mode = pool_mode
        self.out_dim = out_dim
        self.embedding = nn.Embedding(dict_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, feature_dim, kernel_size=conv_k_size)
        if (pool_mode == 'maxpool'):
            self.pool = nn.MaxPool1d(full_vector_size - conv_k_size + 1)
        else:
            self.pool = nn.AvgPool1d(full_vector_size - conv_k_size + 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(feature_dim, out_dim)

    def init_fc_weight(self):
        nn.init.normal_(self.fc.weight, 0.0, 1/math.sqrt(self.fc.weight.size(1)))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # x ~ (batch_size, full_vector_size)
        x1 = self.embedding(x)
        # x1 ~ (batch_size, full_vector_size, hidden_dim)
        x2 = x1.permute(0, 2, 1)
        # x2 ~ (batch_size, hid_dim, full_vector_size)
        x3 = self.conv(x2)
        # x3 ~ (batch_size, feature_dim, full_vector_size - conv_k_size + 1)
        x4 = self.pool(x3)
        # x4 ~ (batch_size, feature_dim, 1)
        x5 = self.relu(x4)
        # x5 ~ (batch_size, feature_dim, 1)
        x6 = x5.view(-1, self.feature_dim)
        # x6 ~ (batch_size, feature_dim)
        x7 = self.fc(x6)
        # x7 ~ (batch_size, 1)
        return torch.sigmoid(x7)


def buildup_dictionary(file_name, with_label = True):
    f = open(file_name)
    dict = {}
    index = 0
    for line in f:
        if with_label:
            parsor = line[2:].split()
        else:
            parsor = line.split()
        for word in parsor:
            if not word in dict:
                dict[word] = index
                index += 1
    print("Dictionary built.")
    return dict


def train(x, y, net, criterion, optimizer, epoch_num, batch_size):

    for epoch in range(epoch_num):
        start = time.time()
        for iter in range(len(x) // batch_size):
            batch_idx = np.random.choice(len(x), batch_size, replace=False)
            x_batch = x[batch_idx, :]
            y_batch = y[batch_idx]

            optimizer.zero_grad()

            scores = net(x_batch)
            running_loss = criterion(scores, y_batch.float())

            running_loss.backward()
            optimizer.step()

            if iter % 10 == 0 or iter == len(x) // batch_size:
                end = time.time()
                print("[ Epoch %d, iteration %4d]: \n    - loss:  %.6f \n    - time: %.3f" %
                      (epoch+1, iter+1, running_loss.item()/batch_size, end-start))
                start = time.time()
    print("Traing ended.")

def run(x, net):
    return net(x)

def test(x, y, net):
    scores = net(x)
    correct = 0
    for i in range(len(scores)):
        correct += int((scores[i] > 0.5) == (y[i] == 1))
    return scores, correct

def load_data(file_name, dictionary, with_label = True):
    f = open(file_name)
    x = []
    y = []
    len_dict = len(dictionary)
    for line in f:
        if with_label:
            y.append(int(line[0]))
        if with_label:
            parsor = line[2:].split()
        else:
            parsor = line.split()
        vec = []
        count = 0
        for word in parsor:
            count += 1
            if word in dictionary:
                vec.append(dictionary[word])
            else:
                vec.append(len_dict)
            if count == full_vector_size:
                break
        for i in range(0, full_vector_size - count):
            vec.append(len_dict + 1)
        x.append(vec)
    print("File ", file_name, " loaded")
    return x, np.asarray(y).reshape(-1, 1)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

full_vector_size = 15

dict = buildup_dictionary("data/train.txt")
x_train, y_train = load_data("data/train.txt", dict)
x_test, y_test = load_data("data/test.txt", dict)
x_unlabel, _ = load_data("data/unlabelled.txt", dict, with_label=False)

x_train = torch.tensor(x_train).to(device)
y_train = torch.tensor(y_train).to(device)
x_test = torch.tensor(x_test).to(device)
y_test = torch.tensor(y_test).to(device)
x_unlabel = torch.tensor(x_unlabel).to(device)

dict_dim = len(dict) + 2
hidden_dim = 300
feature_dim = 128
conv_k_size = 7
pool_mode = 'avgpool' # 'maxpool' or 'avgpool'
out_dim = 1

cnnNet = CNN(dict_dim, hidden_dim, feature_dim, conv_k_size, pool_mode, out_dim).to(device)
cnnNet.init_fc_weight()
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(cnnNet.parameters(), lr = 0.01)

epoch_num = 25
batch_size = 1000

train(x_train, y_train, cnnNet, criterion, optimizer, epoch_num, batch_size)

scores, correct = test(x_test, y_test, cnnNet)
print("Checking accuracy: %f %%" % (100*correct/len(scores)))


