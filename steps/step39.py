import numpy as np
from dozero import Variable
import dozero.functions as F
from dozero.utils import sum_to


x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x.cleargrad()
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
y = x.sum(keepdims=True)
print(y.shape)

# x = np.array([[1, 2, 3], [4, 5, 6]])
# y = sum_to(x, (1, 3))
# print(y)
#
# y = sum_to(x, (2, 1))
# print(y)
