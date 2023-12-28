import numpy as np
from dozero import test_mode
import dozero.functions as F


x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
