from cnn import *
from solver import *
import pickle
from load_MNIST import *


fileTrainSet = 'train-images-idx3-ubyte'
fileTrainLabel = 'train-labels-idx1-ubyte'
fileTestSet = 't10k-images-idx3-ubyte'
fileTestLabel = 't10k-labels-idx1-ubyte'


TrainSet, data_head1 = loadImageSet(fileTrainSet)
TrainLabel, data_head2 = loadLabelSet(fileTrainLabel)
TestSet, data_head3 = loadImageSet(fileTestSet)
TestLabel, data_head4 = loadLabelSet(fileTestLabel)

X_train = np.reshape(TrainSet[0: 55000, :], (55000, 1, 28, 28))
y_train = TrainLabel[0:55000]
X_val = np.reshape(TrainSet[55000: 60000, :], (5000, 1, 28, 28))
y_val = TrainLabel[55000:60000]
X_test = np.reshape(TestSet, (10000, 1, 28, 28))
y_test = TestLabel

data = {
    'X_train':  X_train,
    'y_train':  y_train,
    'X_val':    X_val,
    'y_val':    y_val
}
model = ConvNet(input_dim=(1, 28, 28), hidden_dim = 600, reg = 0, num_classes=10, weight_scale=1e-2)
controller = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={
                    'learning_rate': 1e-1,
                },
                    lr_decay=0.8,
                    num_epochs=20, batch_size=50,
                    print_every=1)
controller.train()
checkAcc = controller.check_accuracy(X_test, y_test)
print("\n")
print("The final testing accuracy is: ", checkAcc)