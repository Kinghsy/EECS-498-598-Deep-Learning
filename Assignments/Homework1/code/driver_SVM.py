from svm import *
from solver import *
import pickle





with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

X_train = data[0][:500]
X_val = data[0][500:750]
X_test = data[0][750:1000]

y_train = data[1][:500].reshape(500)
y_val = data[1][500:750].reshape(250)
y_test = data[1][750:1000].reshape(250)

data = {
    'X_train':  X_train,
    'y_train':  y_train,
    'X_val':    X_val,
    'y_val':    y_val
}
model = SVM(input_dim=20, hidden_dim = 120, reg = 0, weight_scale=1e-2)
controller = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={
                    'learning_rate': 1e0,
                },
                    lr_decay=0.95,
                    num_epochs=200, batch_size=50,
                    print_every=100)
controller.train()
checkAcc = controller.check_accuracy(X_test, y_test)
print("\n")
print("The final testing accuracy is: ", checkAcc)