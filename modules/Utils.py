import time
from functools import wraps
import numpy as np


def crandn(*arg, rng=np.random.random.__self__):
    #np.random.seed()
    return np.sqrt(0.5) * (rng.randn(*arg) + 1j * rng.randn(*arg))


def timethis(func):
    """A decorator that prints the execution time.

    Example:
        Write @utils.timethis before a function definition:

        @utils.timthis
        def my_function():
            pass

        Then, every time my_function is called, the execution time is printed.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        # hours
        h = (toc - tic) // (60 * 60)
        s = (toc - tic) % (60 * 60)
        print(
            'elapsed time of {}(): '
            '{:.0f} hour(s) | {:.0f} minute(s) | {:.5f} second(s).'
            .format(func.__name__, h, s // 60, s % 60)
        )
        return result
    return wrapper
