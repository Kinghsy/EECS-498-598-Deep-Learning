import pickle
import matplotlib.pyplot as plt

loss_his = []
lh = []
loss_his = pickle.load( open('./loss_his.txt', 'rb'))
for i in range(len(loss_his)):
    if i % 10 == 0:
        lh.append(loss_his[i])

plt.plot(lh)
plt.show()