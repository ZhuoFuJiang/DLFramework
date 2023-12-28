import numpy as np
from dozero.layers import Layer
from dozero.core import Parameter
from dozero.layers import Linear
import dozero.functions as F
from dozero.models import Model, MLP
from dozero import optimizers
import matplotlib.pyplot as plt



np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)



lr = 0.2
iters = 10000
hidden_size = 10


model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)


for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 100 == 0:
        print(loss)


fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)
ax.scatter(x, y, label='true')
ax.scatter(x, y_pred.data, color='r', label='pred')
plt.legend()
plt.show()