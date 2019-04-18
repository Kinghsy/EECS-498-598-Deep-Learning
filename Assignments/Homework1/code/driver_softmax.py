from softmax import *
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


data = {
    'X_train':  TrainSet[0: 55000, :],
    'y_train':  TrainLabel[0: 55000],
    'X_val':    TrainSet[55000: 60000, :],
    'y_val':    TrainLabel[55000: 60000]
}
model = SoftmaxClassifier(input_dim=28*28, hidden_dim = 600, reg = 0, num_classes=10, weight_scale=1e-3)
controller = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={
                    'learning_rate': 1e-3,
                },
                    lr_decay=0.8,
                    num_epochs=4, batch_size=50,
                    print_every=100)
controller.train()
checkAcc = controller.check_accuracy(TestSet, TestLabel)
print("\n")
print("The final testing accuracy is: ", checkAcc)