import torch
import numpy as np
from math import sqrt
import torch.nn as nn
import torch.utils.data as data_utils
import time
import pickle
import bcolz

# The GloVe file is downloaded from https://nlp.stanford.edu/data/glove.6B.zip

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
            if count == 10:
                break

        while count < 10:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 10:
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
            if count == 10:
                break

        while count < 10:
            for item in line:
                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 10:
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
            if count == 10:
                break

        while count < 10:
            for item in line:

                if item in dictionary:
                    temp.append(dictionary[item])
                    count += 1
                if count == 10:
                    break

        X_unlabelled.append(temp)


    return X_train, y_train, X_test, y_test, X_unlabelled

def create_emb_layer(weights_matrix, non_trainable=False):
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class q6_4(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()

        self.embed, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)


        self.rnn = nn.RNN(embedding_dim,2)
        self.fc = nn.Linear(20, 1)


    def init_weights(self):
        C_in = self.fc.weight.size(1)
        nn.init.normal_(self.fc.weight, 0.0, 1 / sqrt(C_in))
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        z = self.embed(x.long())
        z, hidden = self.rnn(z)
        z = z.reshape((z.shape[0], z.shape[1]*z.shape[2]))
        z = self.fc(z)
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



    # The GloVe file is downloaded from https://nlp.stanford.edu/data/glove.6B.zip
    # The following usage of GloVe follows the tutorial from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'glove_path/6B.50.dat', mode='w')

    with open(f'glove_path/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'glove_path/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'glove_path/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'glove_path/6B.50_idx.pkl', 'wb'))

    vectors = bcolz.open(f'glove_path/6B.50.dat')[:]
    words = pickle.load(open(f'glove_path/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'glove_path/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(dictionary)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(dictionary):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))



    net = q6_4(weights_matrix).to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)

    f = open('output/predictions_q4.txt', 'w')

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

