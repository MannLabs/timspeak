
# builtin
import dataclasses

# external
import numpy as np
import alphatims.dia_data

# local
import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.indexing
import timspeak.peak_picker_algorithms.cluster.clusters_stats


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class KolmogorovSmirnovTester:
    """
    Kolmogorov-Smirnov Tester class.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index3d (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        precursor_index (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        cluster3d_stats (timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D): The Clusters3D object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    dia_data: alphatims.dia_data.DiaData
    index3d: timspeak.data_handlers.indexing.SparseIndex
    precursor_index: timspeak.data_handlers.indexing.SparseIndex
    cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
    expanded_index_pointers: np.ndarray

    def __post_init__(self):
        pass

    def between_all_clusters(
            self,
            isotopic_pointers,
    ):
        """
        Calculate the Kolmogorov-Smirnov (KS) test between all clusters.

        Parameters:
            isotopic_pointers: The isotopic pointers.

        Returns:
            tuple: A tuple containing:
                lower_clusters: Numpy array of lower clusters.
                upper_clusters: Numpy array of upper clusters.
                ks_values: Numpy array of KS values.
                p_values: Numpy array of p-values.
        """
        lower_isotopes = np.flatnonzero(isotopic_pointers != -1)
        upper_isotopes = isotopic_pointers[lower_isotopes]
        lower_clusters = self.precursor_index.values[lower_isotopes]
        upper_clusters = self.precursor_index.values[upper_isotopes]
        ks_values = np.empty(
            len(lower_isotopes),
            dtype=np.float32,
        )
        p_values = np.empty(
            len(lower_isotopes),
            dtype=np.float32,
        )
        timspeak.performance_utilities.multiprocessing.parallel(self.between_clusters_)(
            range(len(ks_values)),
            lower_clusters,
            upper_clusters,
            ks_values,
            p_values,
        )
        return (
            lower_clusters,
            upper_clusters,
            ks_values,
            p_values,
        )


    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def between_clusters_(
            self,
            precursor_index,
            lower_cluster,
            upper_cluster,
            ks_values,
            p_values,
    ):
        (
            probabilities1,
            probabilities2,
            ks_stat,
            p_value
        ) = self.between_clusters(
            lower_cluster[precursor_index],
            upper_cluster[precursor_index],
        )
        ks_values[precursor_index] = ks_stat
        p_values[precursor_index] = p_value


    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def between_clusters(
            self,
            cluster1,
            cluster2,
    ):
        """
        Calculate the KS test between two clusters.

        Parameters:
            cluster1: The first cluster.
            cluster2: The second cluster.

        Returns:
            tuple: A tuple containing:
                probabilities1: Numpy array of probabilities for cluster 1.
                probabilities2: Numpy array of probabilities for cluster 2.
                ks_stat: The KS statistic.
                p_value: The p-value.
        """
        lower_im_index = min(
            self.cluster3d_stats.im_lower_boundaries[cluster1],
            self.cluster3d_stats.im_lower_boundaries[cluster2],
        )
        upper_im_index = max(
            self.cluster3d_stats.im_upper_boundaries[cluster1],
            self.cluster3d_stats.im_upper_boundaries[cluster2],
        )
        lower_rt_index = min(
            self.cluster3d_stats.rt_lower_boundaries[cluster1],
            self.cluster3d_stats.rt_lower_boundaries[cluster2],
        )
        upper_rt_index = max(
            self.cluster3d_stats.rt_upper_boundaries[cluster1],
            self.cluster3d_stats.rt_upper_boundaries[cluster2],
        )
        probabilities1 = np.zeros(
            (
                (upper_rt_index - lower_rt_index) + 1,
                (upper_im_index - lower_im_index) + 1,
            ),
            dtype=np.float32,
        )
        probabilities2 = np.zeros(
            (
                (upper_rt_index - lower_rt_index) + 1,
                (upper_im_index - lower_im_index) + 1,
            ),
            dtype=np.float32,
        )
        gen1 = self.index3d.generate_from_index(cluster1)
        gen2 = self.index3d.generate_from_index(cluster2)
        for elem1 in gen1:
            index_pointer1 = self.expanded_index_pointers[elem1]
            im_index1 = index_pointer1 % len(self.dia_data.im_values) - lower_im_index
            rt_index1 = index_pointer1 // len(self.dia_data.im_values) - lower_rt_index
            probabilities1[rt_index1, im_index1] += self.dia_data.intensity_values[elem1]
        for r in range(probabilities1.shape[0]):
            probabilities1[r] = np.cumsum(probabilities1[r])
        for r in range(probabilities1.shape[1]):
            probabilities1[:,r] = np.cumsum(probabilities1[:,r])
        probabilities1 /= self.cluster3d_stats.intensity_values[cluster1]
        for elem2 in gen2:
            index_pointer2 = self.expanded_index_pointers[elem2]
            im_index2 = index_pointer2 % len(self.dia_data.im_values) - lower_im_index
            rt_index2 = index_pointer2 // len(self.dia_data.im_values) - lower_rt_index
            probabilities2[rt_index2, im_index2] += self.dia_data.intensity_values[elem2]
        for r in range(probabilities2.shape[0]):
            probabilities2[r] = np.cumsum(probabilities2[r])
        for r in range(probabilities2.shape[1]):
            probabilities2[:,r] = np.cumsum(probabilities2[:,r])
        probabilities2 /= self.cluster3d_stats.intensity_values[cluster2]
        ks_stat = np.max(np.abs(probabilities1 - probabilities2))
        x_size = self.cluster3d_stats.sizes[cluster1]
        y_size = self.cluster3d_stats.sizes[cluster2]
        p_value = 2 * np.exp(
            -ks_stat**2 / (x_size + y_size) * (2 * x_size * y_size)
        )
        return probabilities1, probabilities2, ks_stat, p_value


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class XICCreator:
    """
    XIC Creator class.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index3d (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        cluster3d_stats (timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D): The Clusters3D object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    dia_data: alphatims.dia_data.DiaData
    index3d: timspeak.data_handlers.indexing.SparseIndex
    cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
    expanded_index_pointers: np.ndarray

    def __post_init__(self):
        pass

    def create_xics(self):
        """
        Create XICs (extracted ion chromatograms).

        Returns:
            tuple: A tuple containing:
                xics: Numpy array of XICs.
                xic_indptr: Numpy array of XIC indices.
        """
        xic_indptr = np.empty(len(self.cluster3d_stats) + 1, dtype=np.int64)
        xic_indptr[0] = 0
        xic_indptr[1:] = (
                                 self.cluster3d_stats.rt_upper_boundaries - self.cluster3d_stats.rt_lower_boundaries
                         ) // self.dia_data.cycle.shape[1] + 1
        xic_indptr = np.cumsum(xic_indptr)
        xics = np.zeros(xic_indptr[-1])
        timspeak.performance_utilities.multiprocessing.parallel(self.create_xic_per_cluster)(
            range(len(self.cluster3d_stats)),
            xic_indptr,
            xics,
        )
        return xics, xic_indptr

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def create_xic_per_cluster(
            self,
            cluster_index,
            xic_indptr,
            xics,
    ):
        """
        Create XICs per cluster.

        Parameters:
            cluster_index: The cluster index.
            xic_indptr: Numpy array of XIC indices.
            xics: Numpy array of XICs.
        """
        frame_min = self.cluster3d_stats.rt_lower_boundaries[cluster_index]
        start = xic_indptr[cluster_index]
        end = xic_indptr[cluster_index + 1]
        cycle_xic = xics[start: end]
        for i in self.index3d.generate_from_index(cluster_index):
            intensity = self.dia_data.intensity_values[i]
            indptr = self.expanded_index_pointers[i]
            frame = indptr // self.dia_data.cycle.shape[2] - frame_min
            cycle = frame // self.dia_data.cycle.shape[1]
            cycle_xic[cycle] += intensity
        cycle_xic[:] = np.cumsum(cycle_xic)
        cycle_xic[:] /= cycle_xic[-1]




@timspeak.performance_utilities.compiling.njit
def ks_test_between_cdfs_(
        cdf1,
        cdf2,
        frame_min1,
        frame_min2,
        shape,
        threshold=1
):
    """
    Perform the KS test between two cumulative distribution functions (CDFs).

    Parameters:
        cdf1: The first CDF.
        cdf2: The second CDF.
        frame_min1: The minimum frame for the first CDF.
        frame_min2: The minimum frame for the second CDF.
        shape: The shape of the CDFs.
        threshold: The threshold value (default: 1).

    Returns:
        float: The result of the KS test.
    """
    if frame_min1 < frame_min2:
        cdf1 = cdf1[(frame_min2 - frame_min1) // shape:]
    else:
        cdf2 = cdf2[(frame_min1 - frame_min2) // shape:]
    if len(cdf1) > len(cdf2):
        cdf1 = cdf1[:len(cdf2)]
    else:
        cdf2 = cdf2[:len(cdf1)]
    if len(cdf1) == 0:
        return 1
    m = 0
    for v1, v2 in zip(cdf1, cdf2):
        diff = v1 - v2
        if diff < 0:
            diff = -diff
        if diff > threshold:
            return threshold
        if diff > m:
            m = diff
    return m


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=True)
class MobilogramCreator:
    """
    Mobilogram Creator class.

    Parameters:
        dia_data (alphatims.dia_data.DiaData): The DiaData object.
        index3d (timspeak.data_handlers.indexing.SparseIndex): The SparseIndex object.
        cluster3d_stats (timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D): The Clusters3D object.
        expanded_index_pointers (np.ndarray): Numpy array of expanded index pointers.
    """

    dia_data: alphatims.dia_data.DiaData
    index3d: timspeak.data_handlers.indexing.SparseIndex
    cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
    expanded_index_pointers: np.ndarray

    def __post_init__(self):
        pass

    def create_mobilograms(self):
        """
        Create mobilograms.

        Returns:
            tuple: A tuple containing:
                mobilograms: Numpy array of mobilograms.
                mobilogram_indptr: Numpy array of mobilogram indices.
        """
        mobilogram_indptr = np.empty(len(self.cluster3d_stats) + 1, dtype=np.int64)
        mobilogram_indptr[0] = 0
        mobilogram_indptr[1:] = (
                                        self.cluster3d_stats.im_upper_boundaries - self.cluster3d_stats.im_lower_boundaries
                                ) + 1
        mobilogram_indptr = np.cumsum(mobilogram_indptr)
        mobilograms = np.zeros(mobilogram_indptr[-1])
        timspeak.performance_utilities.multiprocessing.parallel(self.create_mobilogram_per_cluster)(
            range(len(self.cluster3d_stats)),
            mobilogram_indptr,
            mobilograms,
        )
        return mobilograms, mobilogram_indptr

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def create_mobilogram_per_cluster(
            self,
            cluster_index,
            mobilogram_indptr,
            mobilograms,
    ):
        """
        Create mobilograms per cluster.

        Parameters:
            cluster_index: The cluster index.
            mobilogram_indptr: Numpy array of mobilogram indices.
            mobilograms: Numpy array of mobilograms.
        """
        scan_min = self.cluster3d_stats.im_lower_boundaries[cluster_index]
        start = mobilogram_indptr[cluster_index]
        end = mobilogram_indptr[cluster_index + 1]
        cycle_mobilogram = mobilograms[start: end]
        for i in self.index3d.generate_from_index(cluster_index):
            intensity = self.dia_data.intensity_values[i]
            indptr = self.expanded_index_pointers[i]
            scan = indptr % self.dia_data.cycle.shape[2] - scan_min
            cycle_mobilogram[scan] += intensity
        cycle_mobilogram[:] = np.cumsum(cycle_mobilogram)
        cycle_mobilogram[:] /= cycle_mobilogram[-1]
