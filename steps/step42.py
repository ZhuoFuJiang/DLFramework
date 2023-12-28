import numpy as np
from dozero import Variable
import dozero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    # loss = mean_squared_error(y, y_pred)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward(create_graph=True)

    # 梯度下降法
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    # 牛顿迭代法
    # gW = W.grad
    # W.cleargrad()
    # gW.backward(create_graph=True)
    # gW2 = W.grad
    # W.data -= gW.data / gW2.data
    #
    # gb = b.grad
    # b.cleargrad()
    # gb.backward(create_graph=True)
    # gb2 = b.grad
    # b.data -= gb.data / gb2.data
    print(W, b, loss)
