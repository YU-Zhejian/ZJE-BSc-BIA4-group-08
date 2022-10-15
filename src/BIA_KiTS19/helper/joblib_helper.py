__all__ = (
    "parallel_map"
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
    return joblib.Parallel(
        n_jobs=n_jobs, backend=backend
    )(
        joblib.delayed(f)(i) for i in input_iterable
    )
