# builtin
import dataclasses

# external
import numpy as np

# local
import timspeak.performance_utilities.compiling
import timspeak.data_handlers.indexing

import alphatims.dia_data


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class MZIonPairGenerator:
    """
    Generates ion pairs based on TOF indices and tolerance.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object containing TOF indices and mz values.
        ppm_tolerance (float, optional): The tolerance in parts per million (ppm) for mz values. Defaults to 50.0.
        tof_index_tolerance (np.ndarray): The calculated TOF index tolerances.
    """

    dia_data: alphatims.dia_data.DiaData
    ppm_tolerance: float = 50.0
    tof_index_tolerance: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        tof_index_tolerance = self.__calculate_tof_index_tolerances()
        object.__setattr__(self, "tof_index_tolerance", tof_index_tolerance)

    def __calculate_tof_index_tolerances(self) -> np.ndarray:
        tof_index_tolerance = np.searchsorted(
            self.dia_data.mz_values,
            self.dia_data.mz_values * (1 + self.ppm_tolerance * 10**-6),
            ) - np.arange(len(self.dia_data.mz_values))
        return tof_index_tolerance

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def from_scan_pair(
            self,
            scan_index1: int,
            scan_index2: int,
    ) -> (tuple[int, int]):
        """
        Generates ion pairs based on scan indices.
        """
        start1 = self.dia_data.tof_indptr[scan_index1]
        end1 = self.dia_data.tof_indptr[scan_index1 + 1]
        start2 = self.dia_data.tof_indptr[scan_index2]
        end2 = self.dia_data.tof_indptr[scan_index2 + 1]
        for index1 in range(start1, end1):
            tof1 = self.dia_data.tof_indices[index1]
            for index2 in range(start2, end2):
                tof2 = self.dia_data.tof_indices[index2]
                if tof2 > (tof1 + self.tof_index_tolerance[tof1]):
                    break
                if tof1 > (tof2 + self.tof_index_tolerance[tof2]):
                    start2 += 1
                else:
                    yield index1, index2
            else:
                break


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class MZIsotopicClusterPairGenerator(MZIonPairGenerator):
    """
    Generates ion pairs for isotopic clusters based on TOF indices, mz values, and cluster statistics.

    Parameters:
        index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object containing scan index information.
        cluster_stats (np.ndarray): The cluster statistics.
        charge (int, optional): The charge state of the ions. Defaults to 2.
        isotopic_distance (float, optional): The isotopic distance between ions. Defaults to 1.00286864.
    """

    index: timspeak.data_handlers.indexing.SparseIndex
    cluster_stats: np.ndarray
    charge: int = 2
    isotopic_distance: float = 1.00286864

    def __post_init__(self):
        tof_lower_index_tolerance = self.__calculate_tof_lower_index_tolerances()
        object.__setattr__(self, "tof_lower_index_tolerance", tof_lower_index_tolerance)
        tof_upper_index_tolerance = self.__calculate_tof_upper_index_tolerances()
        object.__setattr__(self, "tof_upper_index_tolerance", tof_upper_index_tolerance)

    def __calculate_tof_lower_index_tolerances(self) -> np.ndarray:
        lower_limit = (self.dia_data.mz_values + self.isotopic_distance / self.charge) * (1 - self.ppm_tolerance * 1e-6)
        tof_lower_index_tolerance = np.searchsorted(self.dia_data.mz_values, lower_limit)
        return tof_lower_index_tolerance

    def __calculate_tof_upper_index_tolerances(self) -> np.ndarray:
        upper_limit = (self.dia_data.mz_values + self.isotopic_distance / self.charge) * (1 + self.ppm_tolerance * 1e-6)
        tof_upper_index_tolerance = np.searchsorted(self.dia_data.mz_values, upper_limit)
        return tof_upper_index_tolerance

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def from_scan_pair(
         self,
         scan_index1: int,
         scan_index2: int,
    ) -> (tuple[int, int]):
        """
        Generates ion pairs based on scan indices for isotopic clusters.
        """
        start1 = self.index.indptr[scan_index1]
        end1 = self.index.indptr[scan_index1 + 1]
        start2 = self.index.indptr[scan_index2]
        end2 = self.index.indptr[scan_index2 + 1]
        for index1 in range(start1, end1):
            precursor1 = self.index.values[index1]
            apex1 = self.cluster_stats.apex_indices[precursor1]
            tof1 = self.dia_data.tof_indices[apex1]
            lower_index_tolerance = self.tof_lower_index_tolerance[tof1]
            upper_index_tolerance = self.tof_upper_index_tolerance[tof1]
            for index2 in range(start2, end2):
                precursor2 = self.index.values[index2]
                apex2 = self.cluster_stats.apex_indices[precursor2]
                tof2 = self.dia_data.tof_indices[apex2]
                if tof2 > upper_index_tolerance:
                    break
                if tof2 < lower_index_tolerance:
                    start2 += 1
                else:
                    yield index1, index2
            else:
                break


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class ImNeighborGenerator:
    """
    Generates neighboring scan indices based on ion mobility (IM) values and tolerance.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object containing IM values.
        im_tolerance (float, optional): The tolerance for IM values. Defaults to 0.02.
        scan_index_tolerance (np.ndarray): The calculated scan index tolerances.
    """

    dia_data: alphatims.dia_data.DiaData
    im_tolerance: float = 0.02
    scan_index_tolerance: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        scan_index_tolerance = self.__calculate_scan_index_tolerances()
        object.__setattr__(self, "scan_index_tolerance", scan_index_tolerance)
        object.__setattr__(self, "scans_per_frame", self.dia_data.cycle.shape[-2])

    def __calculate_scan_index_tolerances(self) -> np.ndarray:
        lower_index_tolerance = np.searchsorted(
            -(self.dia_data.im_values),
            -(self.dia_data.im_values + self.im_tolerance),
            "left"
        ) - np.arange(len(self.dia_data.im_values))
        upper_index_tolerance = np.searchsorted(
            -(self.dia_data.im_values),
            -(self.dia_data.im_values - self.im_tolerance),
            "right"
        ) - np.arange(len(self.dia_data.im_values))
        return np.vstack(
            [
                lower_index_tolerance,
                upper_index_tolerance,
            ]
        ).T

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def from_scan(
            self,
            scan_index: int,
    ) -> (int):
        """
        Generates neighboring scan indices based on a given scan index.
        """
        im_index = scan_index % self.scans_per_frame
        im_lower, im_upper = self.scan_index_tolerance[im_index]
        for im_offset in range(im_lower, im_upper):
            new_scan_index = scan_index + im_offset
            yield new_scan_index


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class RtNeighborGenerator:
    """
    Generates neighboring scan indices based on retention time (RT) values and tolerance.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object containing RT values.
        rt_tolerance (float, optional): The tolerance for RT values. Defaults to 3.0.
        frame_index_tolerance (np.ndarray): The calculated frame index tolerances.
    """
    dia_data: alphatims.dia_data.DiaData
    rt_tolerance: float = 3.0
    frame_index_tolerance: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        frame_index_tolerance = self.__calculate_frame_index_tolerances()
        object.__setattr__(self, "frame_index_tolerance", frame_index_tolerance)
        object.__setattr__(self, "scans_per_frame", self.dia_data.cycle.shape[-2])

    def __calculate_frame_index_tolerances(self) -> np.ndarray:
        lower_index_tolerance = np.searchsorted(
            self.dia_data.rt_values,
            self.dia_data.rt_values - self.rt_tolerance,
            "left"
        ) - np.arange(len(self.dia_data.rt_values))
        upper_index_tolerance = np.searchsorted(
            self.dia_data.rt_values,
            self.dia_data.rt_values + self.rt_tolerance,
            "right"
        ) - np.arange(len(self.dia_data.rt_values))
        return np.vstack(
            [
                lower_index_tolerance,
                upper_index_tolerance,
            ]
        ).T

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def from_scan(
            self,
            scan_index: int,
    ) -> (int):
        """
        Generates neighboring scan indices based on a given scan index.
        """
        rt_index = scan_index // self.scans_per_frame
        rt_lower, rt_upper = self.frame_index_tolerance[rt_index]
        for rt_offset in range(rt_lower, rt_upper):
            new_scan_index = scan_index + rt_offset * len(self.dia_data.im_values)
            yield new_scan_index


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class CyclicRtNeighborGenerator(RtNeighborGenerator):
    """
    Generates cyclic neighboring scan indices based on retention time (RT) values, tolerance, and cycle step.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object containing RT values.
        rt_tolerance (float, optional): The tolerance for RT values. Defaults to 3.0.
        frame_index_tolerance (np.ndarray): The calculated frame index tolerances.
        rt_step (int): The cycle step for cyclic RT indexing.
    """

    def __post_init__(self):
        super().__post_init__()
        rt_step = self.dia_data.cycle.shape[1]
        object.__setattr__(self, "rt_step", rt_step)
        self.__update_lower_frame_index_tolerance()

    def __update_lower_frame_index_tolerance(self) -> None:
        offsets = self.frame_index_tolerance[:, 0] % - self.rt_step
        self.frame_index_tolerance[:,0] -= offsets

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def from_scan(
            self,
            scan_index: int,
    ) -> (int):
        """
        Generates cyclic neighboring scan indices based on a given scan index.
        """
        rt_index = scan_index // self.scans_per_frame
        rt_lower, rt_upper = self.frame_index_tolerance[rt_index]
        for rt_offset in range(rt_lower, rt_upper, self.rt_step):
            new_scan_index = scan_index + rt_offset * len(self.dia_data.im_values)
            yield new_scan_index
