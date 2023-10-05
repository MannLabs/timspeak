

import dataclasses
import abc
import numpy as np

@dataclasses.dataclass(frozen=True)
class IConfigFileName(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def check_file_name(
            config_file_name: str,
            config_file_extensions: dict
    ) -> str:
        pass


@dataclasses.dataclass(frozen=True)
class IConfigFileContent(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def read_file_content(config_file_name: str) -> dict:
        pass


class IOutputFileContent(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def read_mem_map(
        *,
        file_name: str,
        mmap_name: str
    ) -> None:
        pass

    @abc.abstractmethod
    def record_sample(
        self,
        sample_file_name: str
    ) -> None:
        pass

    @abc.abstractmethod
    def record_package_info(self) -> None:
        pass

    def record_acquisition(
         self,
         dia_data: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_smoothing_data(
        self,
        smoothing_parameters: dict,
        smooth_intensity_values: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_clustering_as_dataframe(
         self,
         cluster_parameters: dict,
         clusters3d_stats: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_clustering_raw_pointers(
         self,
         index_3d: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_clustering_rt_projection(
         self,
         clusters3d_stats: np.ndarray,
         xics: np.ndarray,
         xic_indptr: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_clustering_im_projection(
         self,
         clusters3d_stats: np.ndarray,
         mobilograms: np.ndarray,
         mobilogram_indptr: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_precursors(
         self,
         precursor_indices: np.ndarray,
         min_size: int
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_2(
         self,
         isotopes_charge_2_parameters: dict,
         lower_isotope_pointers_2: np.ndarray,
         upper_isotope_pointers_2: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_2_metrics(
         self,
         ks_values_rt_2: np.ndarray,
         ks_values_im_2: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_3(
         self,
         isotopes_charge_3_parameters: dict,
         lower_isotope_pointers_3: np.ndarray,
         upper_isotope_pointers_3: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_3_metrics(
         self,
         ks_values_rt_3: np.ndarray,
         ks_values_im_3: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_2_metrics_distance_im_rt(
         self,
         ks_values_2: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_isotopes_charge_3_metrics_distance_im_rt(
         self,
         ks_values_3: np.ndarray
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms1_mono_isotopic_precursors(
         self,
         ks_2d_threshold: float,
         monoisotopic_precursors: np.ndarray,
         monoisotopic_charges: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def print_ms2_fragments(
         self,
         fragment_indices: np.ndarray,
    ) -> None:
        pass