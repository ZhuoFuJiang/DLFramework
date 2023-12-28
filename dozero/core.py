import numpy as np
import weakref
import contextlib
import dozero


try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray, )


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_mode():
    return using_config('train', False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dozero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gxs = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gxs)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def cleargrad(self):
        self.grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = dozero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dozero.cuda.as_cupy(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return "variable(" + p + ")"

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dozero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dozero.functions.transpose(self, axes)

    @property
    def T(self):
        return dozero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dozero.functions.sum(self, axis, keepdims)


class Parameter(Variable):
    pass


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        # 如果列表中只有一个元素，则返回第1个元素
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dozero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dozero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = dozero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dozero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dozero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dozero.functions.sum_to(gx1, self.x1_shape)
        return gx0, -gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        gx0 = gx0 / x1
        gx1 = gx1 * (-x0 / x1 ** 2)

        if self.x0_shape != self.x1_shape:
            gx0 = dozero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dozero.functions.sum_to(gx1, self.x1_shape)

        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1, dozero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


def square(x):
    return Square()(x)


def mul(x0, x1):
    x1 = as_array(x1, dozero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def no_grad():
    return using_config('enable_backprop', False)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


if __name__ == "__main__":
    setup_variable()
    a = Variable(np.array(3.0))
    y = np.array([2.0]) + a
    print(y)
    b = Variable(np.array(2.0))
    # c = Variable(np.array(1.0))
    c = 1.0

    # y = add(mul(a, b), c)
    y = a * b + c
    y.backward(create_graph=True)
    #
    # print(y)
    # print(a.grad)
    # print(b.grad)
    #
    # y = a * b
    # print(y)

    # x = Variable(np.array(2.0))
    # y = -x
    # print(y)
    #
    # y1 = 2.0 - x
    # y2 = x - 1.0
    # print(y1)
    # print(y2)
    #
    # y = x / 0.5
    # print(y)
    #
    # y = x ** 3
    # print(y)

