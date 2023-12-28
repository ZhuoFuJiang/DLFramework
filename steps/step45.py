import numpy as np
from dozero.layers import Layer
from dozero.core import Parameter
from dozero.layers import Linear
import dozero.functions as F
from dozero.models import Model
import matplotlib.pyplot as plt


layer = Layer()

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))
layer.p4 = 'test'

print(layer._params)
print('------------')

for name in layer._params:
    print(name, layer.__dict__[name])


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, inputs):
        y = F.sigmoid(self.l1(inputs))
        y = self.l2(y)
        return y


lr = 0.2
iters = 10000
model = TwoLayerNet(10, 1)
model.plot(x)


for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 100 == 0:
        print(loss)


fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
ax.scatter(x, y, label='true')
ax.scatter(x, y_pred.data, color='r', label='pred')
plt.legend()
plt.show()