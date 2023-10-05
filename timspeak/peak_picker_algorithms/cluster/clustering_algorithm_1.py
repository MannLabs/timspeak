# builtin
import dataclasses

# external
import numpy as np
import alphatims.dia_data

# local
import timspeak.data_handlers.indexing
import timspeak.statistical_utilities.stats
import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.sample_iterator
import timspeak.peak_picker_algorithms.cluster.cluster_templates


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class Clusterer(
    timspeak.peak_picker_algorithms.cluster.cluster_templates.IClusterData
):
    """
    Clusterer class for peak picking and clustering.

    Parameters:
    - dia_data: alphatims.dia_data.DiaData
        The DIA data object.
    - frame_generator: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator
        (Default: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator)
        Frame generator object for cyclic RT neighbor generation.
    - scan_generator: timspeak.data_handlers.sample_iterator.ImNeighborGenerator
        (Default: timspeak.data_handlers.sample_iterator.ImNeighborGenerator)
        Scan generator object for IM neighbor generation.
    - ion_pair_generator: timspeak.data_handlers.sample_iterator.MZIonPairGenerator
        (Default: timspeak.data_handlers.sample_iterator.MZIonPairGenerator)
        Ion pair generator object for MZ ion pair generation.
    - smooth_intensity_values: np.ndarray
        Smoothed intensity values.
    - ppm_tolerance: float
        (Default: 20.0)
        Parts-per-million (PPM) tolerance.
    - im_tolerance: float
        (Default: 0.01)
        IM tolerance.
    - rt_tolerance: float
        (Default: 1.5)
        RT tolerance.
    - clustering_threshold: int
        (Default: 1)
        Clustering threshold.
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
    ion_pair_generator: timspeak.data_handlers.sample_iterator.MZIonPairGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    smooth_intensity_values: np.ndarray
    ppm_tolerance: float = 20.0
    im_tolerance: float = 0.01
    rt_tolerance: float = 1.5
    clustering_threshold: int = 1

    algorithm_name: str = 'clustering_algorithm_1'
    algorithm_description: str = 'This is the original algorithm'

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
        ion_pair_generator = timspeak.data_handlers.sample_iterator.MZIonPairGenerator(
            dia_data = self.dia_data,
            ppm_tolerance=self.ppm_tolerance,
        )
        object.__setattr__(self, "ion_pair_generator", ion_pair_generator)

    def cluster_all_scans(self) -> tuple[np.ndarray]:
        """
        Cluster all scans and return the index 3D array.

        Returns:
        - index_3d: tuple[np.ndarray]
            The index 3D array.
        """
        cluster_pointers = np.arange(len(self.dia_data.intensity_values))
        timspeak.performance_utilities.multiprocessing.parallel(
            self.find_most_intense_neighbors_of_scan
        )(
            range(len(self.dia_data.tof_indptr) - 1),
            cluster_pointers,
        )
        cluster_count = self.update_and_count_cluster_pointers_from_paths(
            cluster_pointers
        )
        indptr, indices = self.update_and_index_cluster_pointers(
            cluster_pointers,
            cluster_count,
        )
        index_3d = timspeak.data_handlers.indexing.SparseIndex(indptr=indptr, values=indices)
        selected_clusters = np.flatnonzero(np.diff(indptr) >= self.clustering_threshold)
        index_3d = index_3d.filter(selected_clusters)
        return index_3d

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def find_most_intense_neighbors_of_scan(
            self,
            scan_index: int,
            most_intense_neighbor_pointers: np.ndarray
    ) -> None:
        """
        Find the most intense neighbors of a scan.

        Parameters:
        - scan_index: int
            The index of the scan.
        - most_intense_neighbor_pointers: np.ndarray
            Array to store the most intense neighbor pointers.
        """
        if self.is_empty_scan(scan_index):
            return
        for other_rt_scan in self.frame_generator.from_scan(scan_index):
            self.get_intense_neighbors_from_scan_generator(
                scan_index,
                other_rt_scan,
                most_intense_neighbor_pointers
            )

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_intense_neighbors_from_scan_generator(
         self,
         scan_index: int,
         other_rt_scan,
         most_intense_neighbor_pointers: np.ndarray
    ) -> None:
        """
        Get the intense neighbors from the scan generator.

        Parameters:
        - scan_index: int
            The index of the scan.
        - other_rt_scan: object
            The other RT scan object.
        - most_intense_neighbor_pointers: np.ndarray
            Array to store the most intense neighbor pointers.
        """
        for other_im_scan in self.scan_generator.from_scan(other_rt_scan):
            if self.is_empty_scan(other_im_scan):
                continue
            self.get_intense_neighbors_from_ion_pair_generator(
                scan_index,
                other_im_scan,
                most_intense_neighbor_pointers
            )

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_intense_neighbors_from_ion_pair_generator(
         self,
         scan_index: int,
         other_im_scan,
         most_intense_neighbor_pointers: np.ndarray
    ) -> None:
        """
        Get the intense neighbors from the ion pair generator.

        Parameters:
        - scan_index: int
            The index of the scan.
        - other_im_scan: object
            The other IM scan object.
        - most_intense_neighbor_pointers: np.ndarray
            Array to store the most intense neighbor pointers.
        """
        for index1, index2 in self.ion_pair_generator.from_scan_pair(
             scan_index,
             other_im_scan
        ):
            pointer = most_intense_neighbor_pointers[index1]
            intensity1 = self.smooth_intensity_values[pointer]
            intensity2 = self.smooth_intensity_values[index2]
            if intensity1 < intensity2:
                most_intense_neighbor_pointers[index1] = index2

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_empty_scan(self, scan_index: int) -> bool:
        """
        Check if a scan is empty.

        Parameters:
        - scan_index: int
            The index of the scan.

        Returns:
        - bool: True if the scan is empty, False otherwise.
        """
        start = self.dia_data.tof_indptr[scan_index]
        end = self.dia_data.tof_indptr[scan_index + 1]
        return start == end

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def update_and_count_cluster_pointers_from_paths(
            self,
            clusters: np.ndarray,
    ) -> int:
        """
        Update and count cluster pointers from paths.

        Parameters:
        - clusters: np.ndarray
            The array of cluster pointers.

        Returns:
        - int: The cluster count.
        """
        cluster_count = 0
        for index, pointer in enumerate(clusters):
            index_pointer = index
            path_length = 1
            while (pointer >= 0) and (index_pointer != pointer):
                index_pointer = pointer
                pointer = clusters[index_pointer]
                path_length += 1
            if pointer >= 0:
                final_pointer = -(cluster_count + 1)
                cluster_count += 1
            else:
                final_pointer = pointer
            for i in range(path_length):
                pointer = clusters[index]
                clusters[index] = final_pointer
                index = pointer
        return cluster_count

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def update_and_index_cluster_pointers(
            self,
            clusters: np.ndarray,
            cluster_count: int,
    ) -> tuple[np.ndarray]:
        """
        Update and index cluster pointers.

        Parameters:
        - clusters: np.ndarray
            The array of cluster pointers.
        - cluster_count: int
            The cluster count.

        Returns:
        - index: tuple[np.ndarray]
            The indptr and indices arrays.
        """
        indptr = np.zeros(cluster_count, dtype=np.int64)
        for index, pointer in enumerate(clusters):
            cluster_index = -(pointer + 1)
            indptr[cluster_index + 1] += 1
            clusters[index] = cluster_index
        indptr = np.cumsum(indptr)
        indptr2 = indptr.copy()
        indices = np.empty_like(clusters)
        for index, pointer in enumerate(clusters):
            spot = indptr2[pointer]
            indices[spot] = index
            indptr2[pointer] += 1
        return indptr, indices

