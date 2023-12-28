import numpy as np
from dozero import Variable
import dozero.functions as F
import matplotlib.pyplot as plt


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


# def predict(x):
#     y = F.linear_simple(x, W1, b1)
#     y = F.sigmoid_simple(y)
#     y = F.linear_simple(y, W2, b2)
#     return y


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000


for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 100 == 0:
        print(loss)


fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
ax.scatter(x, y, label='true')
ax.scatter(x, y_pred.data, color='r', label='pred')
plt.legend()
plt.show()
