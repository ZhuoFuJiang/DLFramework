is_simple_core = False

if is_simple_core:
    from dozero.core_simple import Variable
    from dozero.core_simple import Function
    from dozero.core_simple import using_config
    from dozero.core_simple import no_grad
    from dozero.core_simple import as_array
    from dozero.core_simple import as_variable
    from dozero.core_simple import setup_variable
else:
    from dozero.core import Variable
    from dozero.core import Function
    from dozero.core import using_config
    from dozero.core import no_grad
    from dozero.core import as_array
    from dozero.core import as_variable
    from dozero.core import setup_variable
    from dozero.utils import reshape_sum_backward
    from dozero.functions import Sin
    from dozero.layers import Layer
    from dozero.datasets import Dataset
    from dozero.dataloaders import DataLoader
    from dozero.cuda import get_array_module
    from dozero.core import test_mode
    from dozero.core import Config
    from dozero.models import Model
    from dozero.optimizers import Optimizer

setup_variable()
