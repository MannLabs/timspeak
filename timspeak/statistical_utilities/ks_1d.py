# builtin
import dataclasses

# external
import numpy as np

# local
import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.indexing


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class CDFWithOffset:
    """
    This class represents a cumulative distribution function (CDF) with offsets.
    It is used for handling sparse indices and start offsets.

    Parameters:
    - sparse_indices: An instance of the timspeak.data_handlers.indexing class representing sparse indices.
    - start_offsets: A NumPy array representing the start offsets for the CDF.
    """
    sparse_indices: timspeak.data_handlers.indexing
    start_offsets: np.ndarray

    def __post_init__(self):
        pass

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_valid(self, index: int) -> bool:
        """
        Check if the given index is valid.

        Parameters:
        - index: The index to check.

        Returns:
        - bool: True if the index is valid, False otherwise.
        """
        return self.sparse_indices.is_valid(index)

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_start_offset(self, index: int) -> int:
        """
        Get the start offset for the given index.

        Parameters:
        - index: The index to get the start offset for.

        Returns:
        - int: The start offset value.
        """
        offset = 0
        if self.is_valid(index):
            offset = self.start_offsets[index]
        return offset

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_end_offset(self, index: int) -> int:
        """
        Get the end offset for the given index.

        Parameters:
        - index: The index to get the end offset for.

        Returns:
        - int: The end offset value.
        """
        offset = self.get_start_offset(index)
        offset += self.sparse_indices.get_size(index)
        return offset

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_size(self, index: int) -> int:
        """
        Get the size for the given index.

        Parameters:
        - index: The index to get the size for.

        Returns:
        - int: The size value.
        """
        return self.sparse_indices.get_size(index)

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_cdf(self, index: int):
        """
        Get the cumulative distribution function (CDF) for the given index.

        Parameters:
        - index: The index to get the CDF for.

        Returns:
        - The CDF values.
        """
        return self.sparse_indices.get_values(index)



@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class KSTester1D:
    """
    This class is used for performing the Kolmogorov-Smirnov (KS) test on 1D data.
    It uses a CDF with offsets and a threshold value for the test.

    Parameters:
    - cdf_with_offset: An instance of the CDFWithOffset class representing the CDF with offsets.
    - threshold: The threshold value for the KS test (default is 1.0).
    """

    cdf_with_offset: CDFWithOffset
    threshold: float = 1.0

    def __post_init__(self):
        pass

    def calculate_all(
        self,
        paired_indices: np.ndarray[int, int],
    ) -> np.ndarray[float]:
        """
        Calculate the KS values for all paired indices.

        Parameters:
        - paired_indices: A NumPy array of paired indices for which to calculate the KS values.

        Returns:
        - A NumPy array of KS values.
        """
        ks_values = np.empty(len(paired_indices))
        timspeak.performance_utilities.multiprocessing.parallel(self.calculate_from_buffers)(
            range(len(ks_values)),
            ks_values,
            paired_indices,
        )
        return ks_values

    @timspeak.performance_utilities.compiling.njit
    def calculate_from_buffers(
        self,
        index: int,
        ks_values: np.ndarray[float],
        paired_indices: np.ndarray[int, int],
    ) -> None:
        """
        Calculate the KS value for the given index from buffers.

        Parameters:
        - index: The index for which to calculate the KS value.
        - ks_values: A NumPy array to store the KS values.
        - paired_indices: A NumPy array of paired indices.

        Returns:
        - None
        """
        index1, index2 = paired_indices[index]
        ks_value = self.calculate(
            index1,
            index2,
        )
        ks_values[index] = ks_value

    @timspeak.performance_utilities.compiling.njit
    def calculate(
        self,
        index1: int,
        index2: int,
    ) -> float:
        """
        Calculate the KS value for the given pair of indices.

        Parameters:
        - index1: The first index of the pair.
        - index2: The second index of the pair.

        Returns:
        - The KS value.
        """
        cdf1 = self.cdf_with_offset.get_cdf(index1)
        cdf2 = self.cdf_with_offset.get_cdf(index2)
        start_offset1 = self.cdf_with_offset.get_start_offset(index1)
        start_offset2 = self.cdf_with_offset.get_start_offset(index2)
        max_diff = 0
        if start_offset1 < start_offset2:
            max_diff = cdf1[start_offset2 - start_offset1 - 1]
            cdf1 = cdf1[start_offset2 - start_offset1:]
        elif start_offset2 < start_offset1:
            max_diff = cdf2[start_offset1 - start_offset2 - 1]
            cdf2 = cdf2[start_offset1 - start_offset2:]
        if len(cdf1) > len(cdf2):
            cdf1 = cdf1[:len(cdf2)]
        else:
            cdf2 = cdf2[:len(cdf1)]
        for v1, v2 in zip(cdf1, cdf2):
            diff = v1 - v2
            if diff < 0:
                diff = -diff
            if diff > self.threshold:
                return self.threshold
            if diff > max_diff:
                max_diff = diff
        return max_diff
