import numpy as np
from dozero import Variable


x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

x0.cleargrad()
x1.cleargrad()
y = x0 - x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

x0.cleargrad()
x1.cleargrad()
y = x0 * x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)

x0.cleargrad()
x1.cleargrad()
y = x0 / x1
print(y)
y.backward()
print(x0.grad)
print(x1.grad)
