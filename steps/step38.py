import numpy as np
from dozero import Variable
import dozero.functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6, ))
# y.backward(retain_grad=True)
y.backward()
print(x.grad)


x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
y = x.reshape(2, 3)


y = F.transpose(x)
y.backward()
print(x.grad)

x = Variable(np.array(np.random.randn(2, 3)))
y = x.transpose()
y = x.T
print(y)