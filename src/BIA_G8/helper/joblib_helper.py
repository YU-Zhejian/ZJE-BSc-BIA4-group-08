"""
Here provides wrappers for important :mod:`joblib` modules.
``joblib`` is a Python package that allows easy parallelization using minimal configuration.
"""

__all__ = (
    "parallel_map",
)

import multiprocessing
from typing import Callable, TypeVar, Iterable

import joblib

_InType = TypeVar("_InType")
_OutType = TypeVar("_OutType")


def parallel_map(
        f: Callable[[_InType], _OutType],
        input_iterable: Iterable[_InType],
        n_jobs: int = multiprocessing.cpu_count(),
        backend: str = "loky",
) -> Iterable[_OutType]:
    """
    The parallel version of Python :external:py:func:`map` function (or, ``apply`` function in R).

    See also: :external+joblib:py:class:`joblib.Parallel`.

    .. warning::
        With inappropriate parallelization, the system would consume lots of memory with minimal speed improvement!

    .. warning::
        Use with caution if you wish to parallely assign elements to an array.

    :param f: Function to be applied around an iterable.
    :param input_iterable: Iterable where a function would be applied to.
    :param n_jobs: Number of parallel threads. Would be max available CPU number if not set.
    :param backend: The backend to be used. Recommended to use ``loky``. You may also try ``threading`` if ``loky`` fails.
    :return: Generated new iterable.
    """
    return joblib.Parallel(
        n_jobs=n_jobs, backend=backend
    )(
        joblib.delayed(f)(i) for i in input_iterable
    )
