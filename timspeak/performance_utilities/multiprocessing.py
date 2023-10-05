# builtin
import multiprocessing
import multiprocessing.pool
import functools
import threading

# external
import tqdm
import numba
import numpy as np


MAX_THREADS = multiprocessing.cpu_count()


def set_threads(threads: int, set_global: bool = True) -> int:
    max_cpu_count = multiprocessing.cpu_count()
    if threads > max_cpu_count:
        threads = max_cpu_count
    else:
        while threads <= 0:
            threads += max_cpu_count
    if set_global:
        global MAX_THREADS
        MAX_THREADS = threads
    return threads


def threadpool(
    _func=None,
    *,
    thread_count=None,
    return_results: bool = False,
) -> None:
    import functools

    def parallel_func_inner(func):
        def wrapper(iterable, *args, **kwargs):
            def starfunc(iterable):
                return func(iterable, *args, **kwargs)

            try:
                iter(iterable)
            except TypeError:
                return func(iterable, *args, **kwargs)
            if thread_count is None:
                current_thread_count = MAX_THREADS
            else:
                current_thread_count = set_threads(
                    thread_count,
                    set_global=False
                )
            with multiprocessing.pool.ThreadPool(current_thread_count) as pool:
                if return_results:
                    results = []
                    for result in tqdm.tqdm(
                        pool.imap(starfunc, iterable),
                        total=len(iterable),
                    ):
                        results.append(result)
                    return results
                else:
                    for result in tqdm.tqdm(
                        pool.imap_unordered(starfunc, iterable),
                        total=len(iterable),
                    ):
                        pass
        return functools.wraps(func)(wrapper)
    if _func is None:
        return parallel_func_inner
    else:
        return parallel_func_inner(_func)



def parallel(
    _func=None,
    *,
    thread_count=None,
    include_progress_callback: bool = True,
):
    def parallel_compiled_func_inner(func):
        numba_func = func

        @numba.njit(nogil=True)
        def numba_func_parallel(
            iterable,
            thread_id,
            progress_counter,
            start,
            stop,
            step,
            *args,
        ):
            if len(iterable) == 0:
                for i in range(start, stop, step):
                    numba_func(i, *args)
                    progress_counter[thread_id] += 1
            else:
                for i in iterable:
                    numba_func(i, *args)
                    progress_counter[thread_id] += 1

        def wrapper(iterable, *args):
            if thread_count is None:
                current_thread_count = MAX_THREADS
            else:
                current_thread_count = set_threads(
                    thread_count,
                    set_global=False
                )
            threads = []
            progress_counter = np.zeros(current_thread_count, dtype=np.int64)
            for thread_id in range(current_thread_count):
                local_iterable = iterable[thread_id::current_thread_count]
                if isinstance(local_iterable, range):
                    start = local_iterable.start
                    stop = local_iterable.stop
                    step = local_iterable.step
                    local_iterable = np.array([], dtype=np.int64)
                else:
                    start = -1
                    stop = -1
                    step = -1
                thread = threading.Thread(
                    target=numba_func_parallel,
                    args=(
                        local_iterable,
                        thread_id,
                        progress_counter,
                        start,
                        stop,
                        step,
                        *args
                    ),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
            if include_progress_callback:
                import time
                if len(iterable) > 10**6:
                    granularity = 1000
                else:
                    granularity = len(iterable)
                progress_bar = 0
                progress_count = np.sum(progress_counter)
                for result in tqdm.tqdm(
                    range(granularity),
                ):
                    while progress_bar >= progress_count:
                        time.sleep(0.01)
                        progress_count = granularity * np.sum(progress_counter) / len(iterable)
                    progress_bar += 1
            for thread in threads:
                thread.join()
                del thread
        return functools.wraps(func)(wrapper)
    if _func is None:
        return parallel_compiled_func_inner
    else:
        return parallel_compiled_func_inner(_func)

