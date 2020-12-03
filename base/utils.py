import random
import torch
import numpy
import inspect
import warnings

def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def dispatch_args_to_func(func, argspace, **kwargs):
    func_args = inspect.getfullargspec(func).args
    for x in func_args:
        if x not in argspace:
            warnings.warn("functional argument {} is not in argspace.".format(x), RuntimeWarning)
    args = {x: t for x, t in argspace.items() if x in func_args}
    for i, x in kwargs.items():
        args[i] = x
    return func(**args)

def get_last_valid_column(masks):
    """ Return the index of the column after which, inclusively,
    all columns are all False (invalid).
    """
    has_valid_token = torch.sum(masks, 0) > 0
    ridx = masks.shape[1] - 1
    while ridx >= 0:
        if not has_valid_token[ridx]:
            ridx -= 1
            continue
        break
    return ridx + 1