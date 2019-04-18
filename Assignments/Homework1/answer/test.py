from layers import *
import numpy as np
from optim import *
from logistic import *
from solver import *
import pickle






with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')



X_train = data[0][:500]
X_val = data[0][500:750]
X_test = data[0][750:1000]

y_train = data[1][:500]
y_val = data[1][500:750]
y_test = data[1][750:1000]

data = {
    'X_train':  X_train,
    'y_train':  y_train,
    'X_val':    X_val,
    'y_val':    y_val
}
model = LogisticClassifier(input_dim=20, hidden_dim = 40, reg = 0)
solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={
                    'learning_rate': 1e0
                },
                lr_decay=0.85,
                num_epochs=180, batch_size=50,
                print_every=100)
solver.train()