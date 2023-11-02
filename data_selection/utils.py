from typing import Callable, List, Any
from joblib import Parallel, delayed


def parallelize(fn: Callable, args: List[Any], num_proc: int):
    return Parallel(n_jobs=num_proc)(delayed(fn)(arg) for arg in args)
