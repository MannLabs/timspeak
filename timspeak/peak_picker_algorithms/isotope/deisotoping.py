# builtin
import dataclasses

# external
import numpy as np
import alphatims.dia_data

# local
import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.sample_iterator
import timspeak.data_handlers.indexing
import timspeak.statistical_utilities.stats


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class ChargeDeisotoper:
    """
    ChargeDeisotoper class for charge deisotoping.

    Parameters:
    - dia_data: alphatims.dia_data.DiaData
        The DIA data object.
    - frame_generator: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator
        (Default: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator)
        Frame generator object for cyclic RT neighbor generation.
    - scan_generator: timspeak.data_handlers.sample_iterator.ImNeighborGenerator
        (Default: timspeak.data_handlers.sample_iterator.ImNeighborGenerator)
        Scan generator object for IM neighbor generation.
    - isotope_pair_generator: timspeak.data_handlers.sample_iterator.MZIsotopicClusterPairGenerator
        (Default: timspeak.data_handlers.sample_iterator.MZIsotopicClusterPairGenerator)
        Isotope pair generator object for MZ isotopic cluster pair generation.
    - index: timspeak.data_handlers.indexing.SparseIndex
        The index object.
    - ppm_tolerance: float
        (Default: 20.0)
        Parts-per-million (PPM) tolerance.
    - im_tolerance: float
        (Default: 0.01)
        IM tolerance.
    - rt_tolerance: float
        (Default: 1.5)
        RT tolerance.
    - charge: int
        The charge value.
    - cluster_stats: timspeak.statistical_utilities.stats.StatsCalculator
        The statistics calculator object.
    """

    dia_data: alphatims.dia_data.DiaData
    frame_generator: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    scan_generator: timspeak.data_handlers.sample_iterator.ImNeighborGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    isotope_pair_generator: timspeak.data_handlers.sample_iterator.MZIsotopicClusterPairGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    index: timspeak.data_handlers.indexing.SparseIndex
    ppm_tolerance: float = 20.0
    im_tolerance: float = 0.01
    rt_tolerance: float = 1.5
    charge: int
    cluster_stats: timspeak.statistical_utilities.stats.StatsCalculator

    def __post_init__(self):
        frame_generator = timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator(
            dia_data = self.dia_data,
            rt_tolerance=self.rt_tolerance,
        )
        object.__setattr__(self, "frame_generator", frame_generator)
        scan_generator = timspeak.data_handlers.sample_iterator.ImNeighborGenerator(
            dia_data = self.dia_data,
            im_tolerance=self.im_tolerance,
        )
        object.__setattr__(self, "scan_generator", scan_generator)
        isotope_pair_generator = timspeak.data_handlers.sample_iterator.MZIsotopicClusterPairGenerator(
            dia_data = self.dia_data,
            ppm_tolerance=self.ppm_tolerance,
            charge=self.charge,
            index=self.index,
            cluster_stats=self.cluster_stats,
        )
        object.__setattr__(self, "isotope_pair_generator", isotope_pair_generator)

    def deisotope_all_scans(
        self,
    ) -> tuple[np.ndarray]:
        """
        Deisotope all scans and return the charge pointers.

        Returns:
        - charge_pointers: tuple[np.ndarray]
            The charge pointers array.
        """
        charge_pointers = np.empty(self.index.shape[1], dtype=np.int64)
        charge_pointers[:] = -1
        timspeak.performance_utilities.multiprocessing.parallel(
            self.deisotope_scan
        )(
            range(len(self.index.indptr) - 1),
            charge_pointers,
        )
        return charge_pointers

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def deisotope_scan(
        self,
        scan_index: int,
        charge_pointers: np.ndarray,
    ) -> None:
        """
        Deisotope a single scan.

        Parameters:
        - scan_index: int
            The index of the scan.
        - charge_pointers: np.ndarray
            Array to store the charge pointers.
        """
        if self.is_empty_scan(scan_index):
            return
        for other_rt_scan in self.frame_generator.from_scan(scan_index):
            for other_im_scan in self.scan_generator.from_scan(other_rt_scan):
                if self.is_empty_scan(other_im_scan):
                    continue
                for index1, index2 in self.isotope_pair_generator.from_scan_pair(
                    scan_index,
                    other_im_scan
                ):
                    charge_pointers[index1] = index2

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_empty_scan(self, scan_index: int) -> bool:
        """
        Check if a scan is empty.

        Parameters:
        - scan_index: int
            The index of the scan.

        Returns:
        - bool: True if the scan is empty, otherwise False.
        """
        start = self.index.indptr[scan_index]
        end = self.index.indptr[scan_index + 1]
        return start == end