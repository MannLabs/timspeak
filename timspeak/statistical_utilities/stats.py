# builtin
import abc
import dataclasses

# external
import numpy as np
import alphatims.dia_data

# local
import timspeak.data_handlers.indexing

import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.indexing



@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class StatsCalculator(abc.ABC):
    """
    Abstract base class for statistics calculators.
    Calculate the statistics for all clusters.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        dtype (type): The data type of the calculated statistics (default: np.float32).
        dimension_count (int): The number of dimensions for the statistics (default: 1).
    """

    dia_data: alphatims.dia_data.DiaData
    index: timspeak.data_handlers.indexing.SparseIndex
    dtype: type = np.float32
    dimension_count: int = 1

    def calculate(self,) -> np.ndarray:
        """
        Calculate the statistics for all clusters.

        Returns:
            np.ndarray: The calculated statistics as a numpy array.
        """
        if self.dimension_count > 1:
            shape = (self.index.size, self.dimension_count)
        else:
            shape = self.index.size
        buffer_array = np.empty(
                shape,
            dtype=self.dtype
        )
        timspeak.performance_utilities.multiprocessing.parallel(self._calculate_per_cluster)(
            range(self.index.size),
            buffer_array
        )
        return buffer_array

    @abc.abstractmethod
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        buffer_array: np.ndarray
    ) -> None:
        pass




@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class SizeCalculator(StatsCalculator):
    """
    SizeCalculator class.
    Calculate the size (number of peaks) for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        dtype (type): The data type of the calculated statistics (default: np.int32).
    """

    dtype: type = np.int32

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        intensity_values: np.ndarray
    ) -> None:
        start, end = self.index.get_boundaries(cluster_pointer)
        intensity_values[cluster_pointer] = end - start


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class MzCalculator(StatsCalculator):
    """
    MzCalculator class.
    Calculate the weighted mean m/z value for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
    """
    dtype: type = np.float32

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        mz_values: np.ndarray
    ) -> None:
        summed_mz_value = 0
        summed_intensity_value = 0
        for index in self.index.generate_from_index(cluster_pointer):
            intensity_value = self.dia_data.intensity_values[index]
            summed_intensity_value += intensity_value
            tof_index = self.dia_data.tof_indices[index]
            mz_value = self.dia_data.mz_values[tof_index]
            summed_mz_value += intensity_value * mz_value
        mz_values[cluster_pointer] = summed_mz_value / summed_intensity_value


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class IntensityCalculator(StatsCalculator):
    """
    Intensity Calculator class.
    Calculate the sum of intensities for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
    """

    dtype: type = np.float64

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
            self,
            cluster_pointer: int,
            intensity_values_2: np.ndarray
    ) -> None:
        summed_intensity_value = 0
        for index in self.index.generate_from_index(cluster_pointer):
            intensity_value = self.dia_data.intensity_values[index]
            summed_intensity_value += intensity_value
        intensity_values_2[cluster_pointer] = summed_intensity_value


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class ImCalculator(StatsCalculator):
    """
    ImCalculator class.
    Calculate the weighted mean IM (ion mobility) value for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    expanded_index_pointers: np.ndarray

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        im_values: np.ndarray
    ) -> None:
        summed_im_value = 0
        summed_intensity_value = 0
        for index in self.index.generate_from_index(cluster_pointer):
            intensity_value = self.dia_data.intensity_values[index]
            expanded_index = self.expanded_index_pointers[index]
            im_index = expanded_index % len(self.dia_data.im_values)
            im_value = self.dia_data.im_values[im_index]
            summed_intensity_value += intensity_value
            summed_im_value += intensity_value * im_value
        im_values[cluster_pointer] = summed_im_value / summed_intensity_value


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class RtCalculator(StatsCalculator):
    """
    RtCalculator class.
    Calculate the weighted mean retention time (RT) value for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    expanded_index_pointers: np.ndarray

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        rt_values: np.ndarray
    ) -> None:
        summed_rt_value = 0
        summed_intensity_value = 0
        for index in self.index.generate_from_index(cluster_pointer):
            intensity_value = self.dia_data.intensity_values[index]
            expanded_index = self.expanded_index_pointers[index]
            rt_index = expanded_index // len(self.dia_data.im_values)
            rt_value = self.dia_data.rt_values[rt_index]
            summed_intensity_value += intensity_value
            summed_rt_value += intensity_value * rt_value
        rt_values[cluster_pointer] = summed_rt_value / summed_intensity_value


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class ApexCalculator(StatsCalculator):
    """
    Apex Calculator class.
    Find the index of the peak with the highest intensity for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        smooth_intensity_values (np.ndarray): Numpy array of smooth intensity values.
    """

    smooth_intensity_values: np.ndarray
    dtype: type = np.int64

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        apex_indices: np.ndarray
    ) -> None:
        max_intensity = -np.inf
        for index in self.index.generate_from_index(cluster_pointer):
            intensity_value = self.smooth_intensity_values[index]
            if intensity_value > max_intensity:
                max_intensity = intensity_value
                apex_indices[cluster_pointer] = index


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class IMBoundaryCalculator(StatsCalculator):
    """
    IMBoundaryCalculator class.
    Calculate the minimum and maximum IM boundaries for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.

    """

    expanded_index_pointers: np.ndarray
    dtype: type = np.int32
    dimension_count: int = 2

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        boundaries: np.ndarray
    ) -> None:
        min_im_index = np.inf
        max_im_index = -np.inf
        for index in self.index.generate_from_index(cluster_pointer):
            expanded_index = self.expanded_index_pointers[index]
            im_index = expanded_index % len(self.dia_data.im_values)
            if im_index < min_im_index:
                min_im_index = im_index
            if im_index > max_im_index:
                max_im_index = im_index
        boundaries[cluster_pointer] = (min_im_index, max_im_index)


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class RTBoundaryCalculator(StatsCalculator):
    """
    RTBoundaryCalculator class.
    Calculate the minimum and maximum RT boundaries for each cluster.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    expanded_index_pointers: np.ndarray
    dtype: type = np.int32
    dimension_count: int = 2

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def _calculate_per_cluster(
        self,
        cluster_pointer: int,
        boundaries: np.ndarray
    ) -> None:
        min_rt_index = np.inf
        max_rt_index = -np.inf
        for index in self.index.generate_from_index(cluster_pointer):
            expanded_index = self.expanded_index_pointers[index]
            rt_index = expanded_index // len(self.dia_data.im_values)
            if rt_index < min_rt_index:
                min_rt_index = rt_index
            if rt_index > max_rt_index:
                max_rt_index = rt_index
        boundaries[cluster_pointer] = (min_rt_index, max_rt_index)
