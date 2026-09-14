import time
import traceback
from shutil import get_terminal_size
from multiprocess import Pool, Manager
from typing import Iterable, Callable, TypeVar, Optional, Generic
from rich.console import Console
console = Console()
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)

T = TypeVar('T')
R = TypeVar('R')


class ParallelMapper(Generic[T, R]):
    """
    A class to handle parallel mapping operations with progress bar.

    Attributes:
        max_workers (Optional[int]): Maximum number of worker threads
        description (str): Progress bar description
        progress_config (dict): Progress bar configuration
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        description: str = "Processing",
        use_progress_bar: bool = True,
        progress_config: Optional[dict] = None,
    ):
        self.max_workers = max_workers
        self.description = "[bold red]" + description
        self.progress_config = progress_config or {}
        self.use_progress_bar = use_progress_bar

    def map(self, func: Callable[..., R], *items: Iterable[T]) -> list[R]:
        """
        Maps function over items in parallel with progress tracking.

        Args:
            func: Function to apply to each item
            items: Iterable of input items

        Returns:
            Iterator of results

        Raises:
            Exception: If processing fails
        """
        counter = Manager().Value('Q', 0)
        total = len(list(items[0]))

        def wrapper(*args):
            try:
                result = func(*args)
                counter.value += 1
                return result
            except Exception as e:
                raise RuntimeError(f"\n[ERROR] SUBTASK FAILED:\n\n- Args:{args}\n\n- Error Message:{str(e)}\n")

        def get_results(async_results, counter, total):
            if self.use_progress_bar:
                progress = Progress(
                    SpinnerColumn(finished_text="[green]✓"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=int(get_terminal_size().columns / 2)),
                    TaskProgressColumn("[progress.percentage]{task.percentage:>3.2f}% [{task.completed}/{task.total}]"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    **self.progress_config
                )

                with progress:
                    task = progress.add_task(self.description, total=total)
                    current = 0
                    while not async_results.ready():
                        cnt = counter.value
                        if cnt != current:
                            current = cnt
                            progress.update(task, completed=current)
                        time.sleep(0.1)
                    try:
                        results = list(async_results.get())
                        progress.update(task, completed=total)
                        progress.update(task, description="[bold green]Done")
                        return results
                    except Exception as e:
                        cnt = counter.value
                        if cnt != current:
                            progress.update(task, completed=cnt)
                        progress.update(task, description="[bold red]Failed")
                        raise RuntimeError(f"{e}")
            else:
                def log_progress(current, total):
                    percentage = (current / total) * 100
                    console.log(
                        f"Progress: {percentage:>6.2f}% ({current}/{total})",
                        end='' if current < total else '\n')

                current = 0
                while not async_results.ready():
                    cnt = counter.value
                    if cnt != current:
                        current = cnt
                        log_progress(current, total)
                    time.sleep(0.1)
                try:
                    results = list(async_results.get())
                    log_progress(total, total)
                    return results
                except Exception as e:
                    cnt = counter.value
                    if cnt != current:
                        log_progress(cnt, total)
                    raise RuntimeError(f"{e}")

        with Pool(self.max_workers) as p:
            list_items = list(zip(*items))
            async_results = p.starmap_async(wrapper, list_items)
            results = get_results(async_results, counter, total)

        return results


def parallel_map(
    func: Callable[..., R],
    *items: Iterable[T],
    max_workers: Optional[int] = None,
    description: str = "Processing",
    use_progress_bar: bool = False,
) -> list[R]:
    """Convenience function for parallel mapping with progress bar."""
    mapper = ParallelMapper(max_workers=max_workers,
                            description=description,
                            use_progress_bar=use_progress_bar)
    return mapper.map(func, *items)


if __name__ == "__main__":
    # Test cases
    import time
    import random
    def single_arg_operation(x: int) -> int:
        time.sleep(random.uniform(0.5, 2.0))
        return x * (x * 10)

    def multi_args_operation(x: int, y: int) -> int:
        time.sleep(random.uniform(0.5, 2.0))
        return x * y

    def task_without_return(x: int) -> None:
        time.sleep(random.uniform(0.5, 2.0))
        print(f"Task {x} done: {x * (x * 10)}")

    def task_with_error(x: int) -> int:
        time.sleep(random.uniform(0.5, 2.0))
        if x % 2 == 0:
            raise ValueError(f"Subtask {x} failed")
        return x * (x * 10)

    x = range(100)
    y = range(0, 1000, 10)

    console.log("Test 1: single arg operation")
    results = parallel_map(single_arg_operation, x, max_workers=64, description="Calculating")
    console.log(f"Results: {results}")

    console.log("Test 2: multi args operation")
    results = parallel_map(multi_args_operation, x, y, max_workers=64, description="Calculating")
    console.log(f"Results: {results}")

    console.log("Test 3: task without return")
    parallel_map(task_without_return, x, max_workers=64, description="Calculating")

    console.log("Test 4: task with error")
    try:
        parallel_map(task_with_error, x, max_workers=64, description="Calculating")
    except Exception as e:
        import traceback
        console.log(f"Successfully caught error: {e}")
        console.log(f"Traceback:\n{traceback.format_exc()}")
