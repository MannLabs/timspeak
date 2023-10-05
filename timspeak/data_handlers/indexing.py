
# builtin
import dataclasses

# external
import numpy as np

# local
import timspeak.performance_utilities.compiling


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class SparseIndex:
    """
    SparseIndex class for sparse indexing.

    Parameters:
    - indptr: np.ndarray
        The index pointer array.
    - values: np.ndarray
        The values array.
    - size: int
        (Default: calculated from indptr)
        The size of the index.
    """

    indptr: np.ndarray = dataclasses.field(repr=False)
    values: np.ndarray = dataclasses.field(repr=False)
    size: int = dataclasses.field(init=False, repr=False)
    shape: tuple[int, int] = dataclasses.field(init=False)

    def __post_init__(self):
        size = len(self.indptr) - 1
        object.__setattr__(self, "size", size)
        shape = tuple([self.size, self.indptr[-1]])
        object.__setattr__(self, "shape", shape)

    def __len__(self):
        return self.size

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_values(self, index: int) -> np.ndarray:
        """
        Get the values for the given index.

        Parameters:
        - index: int
            The index.

        Returns:
        - np.ndarray: The values for the given index.
        """
        start, end = self.get_boundaries(index)
        return self.values[start: end]

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_size(self, index: int) -> int:
        """
        Get the size of the values for the given index.

        Parameters:
        - index: int
            The index.

        Returns:
        - int: The size of the values for the given index.
        """
        start, end = self.get_boundaries(index)
        return end - start

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_boundaries(self, index: int) -> tuple[int, int]:
        """
        Get the start and end boundaries for the given index.

        Parameters:
        - index: int
            The index.

        Returns:
        - tuple[int, int]: The start and end boundaries for the given index.
        """
        start = 0
        end = 0
        if self.is_valid(index):
            start = self.indptr[index]
            end = self.indptr[index + 1]
        return start, end

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_valid(self, index: int) -> bool:
        """
        Check if the given index is valid.

        Parameters:
        - index: int
            The index.

        Returns:
        - bool: True if the index is valid, otherwise False.
        """
        return 0 <= index < self.size

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_empty(self, index: int) -> bool:
        """
        Check if the values for the given index are empty.

        Parameters:
        - index: int
            The index.

        Returns:
        - bool: True if the values are empty, otherwise False.
        """
        start, end = self.get_boundaries(index)
        return start == end

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def generate_from_index(self, index):
        """
        Generate values from the given index.

        Parameters:
        - index: int
            The index.

        Yields:
        - The values from the given index.
        """
        start, end = self.get_boundaries(index)
        for offset in range(start, end):
            yield self.values[offset]

    def filter(self, indices: np.ndarray):
        """
        Filter the index using the given indices.

        Parameters:
        - indices: np.ndarray
            The indices to filter the index with.

        Returns:
        - SparseIndex: A new SparseIndex object with the filtered index.
        """
        new_indptr = np.zeros(len(indices) + 1, dtype=self.indptr.dtype)
        new_indptr[1:] = self.indptr[indices + 1] - self.indptr[indices]
        new_indptr = np.cumsum(new_indptr)
        new_values = np.empty(new_indptr[-1], dtype=self.values.dtype)
        timspeak.performance_utilities.multiprocessing.parallel(
            self._set_new_values_after_filtering,
        )(
            range(len(indices)),
            indices,
            new_indptr,
            new_values,
        )
        return type(self)(indptr=new_indptr, values=new_values)

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _set_new_values_after_filtering(
         self,
         index: int,
         indices: np.ndarray,
         new_indptr: np.ndarray,
         new_values: np.ndarray,
    ) -> None:
        old_index = indices[index]
        values = self.get_values(old_index)
        start = new_indptr[index]
        end = new_indptr[index + 1]
        new_values[start: end] = values