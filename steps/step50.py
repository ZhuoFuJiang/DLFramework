import math

import numpy as np
import dozero
from dozero.models import MLP
from dozero import optimizers
from dozero import functions as F


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0


def f(x):
    y = x / 2.0
    return y


train_set = dozero.datasets.Spiral(train=True)
test_set = dozero.datasets.Spiral(train=False)
train_loader = dozero.DataLoader(train_set, batch_size)
test_loader = dozero.DataLoader(test_set, batch_size, shuffle=False)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

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

