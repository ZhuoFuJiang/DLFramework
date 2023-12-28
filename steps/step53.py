import dozero
import matplotlib.pyplot as plt
import numpy as np
import os
from dozero.models import MLP
from dozero import optimizers
from dozero import functions as F

train_set = dozero.datasets.MNIST(train=True, transform=None)
test_set = dozero.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))


# x, t = train_set[0]
# plt.imshow(x.reshape(28, 28), cmap='gray')
# plt.axis('off')
# plt.show()
# print('label:', t)

max_epoch = 5
batch_size = 100
hidden_size = 1000
lr = 1.0


def f(x):
    y = x / 2.0
    return y


train_set = dozero.datasets.MNIST(train=True)
test_set = dozero.datasets.MNIST(train=False)
train_loader = dozero.DataLoader(train_set, batch_size)
test_loader = dozero.DataLoader(test_set, batch_size, shuffle=False)
# model = MLP((hidden_size, 10))
model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)

# 加载参数
if os.path.exists("my_mlp.npz"):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss, sum_acc = 0, 0

    for batch_x, batch_t in train_loader:

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        acc = F.accuracy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
        sum_acc += float(acc.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    avg_acc = sum_acc / data_size
    print('epoch %d, loss %.2f train accuracy %.2f' % (epoch + 1, avg_loss, avg_acc))

    sum_loss, sum_acc = 0, 0
    with dozero.no_grad():
        for batch_x, batch_t in test_loader:
            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            acc = F.accuracy(y, batch_t)
            sum_loss += float(loss.data) * len(batch_t)
            sum_acc += float(acc.data) * len(batch_t)
    avg_loss = sum_loss / len(test_set)
    avg_acc = sum_acc / len(test_set)
    print('epoch %d, loss %.2f test accuracy %.2f' % (epoch + 1, avg_loss, avg_acc))

model.save_weights('my_mlp.npz')
