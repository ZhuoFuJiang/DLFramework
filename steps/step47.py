import numpy as np
from dozero.models import MLP
from dozero import Variable, as_variable
import dozero.functions as F


model = MLP((10, 3))


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax1d(y)
print(y)
print(p)
