import torch
import numpy as np
from math import sqrt
import torch.nn as nn
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



def create_emb_layer(weights_matrix, non_trainable=False):
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim



class q6_3(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()

        self.embed, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)

        self.fc = nn.Linear(embedding_dim, 1)


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

    vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'glove_path/6B.50.dat', mode='w')
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



    net = q6_3(weights_matrix).to(device)
    net.init_weights()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.08)

    train(X_train, y_train, net, criterion, optimizer)
    test(X_test, y_test, net)

    f = open('output/predictions_q3.txt', 'w')

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

