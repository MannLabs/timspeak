
import dataclasses
import numpy as np
import timspeak.io_interface.configuration_templates
import timspeak.io_interface.output.out_formats.zarr_object

@dataclasses.dataclass(frozen=True)
class ZARRFormat(
    timspeak.io_interface.configuration_templates.IOutputFileContent
):

    def __init__(
        self,
        output_file_name: str
    ) -> None:
        object.__setattr__(self, 'output_file_name', output_file_name)
        object.__setattr__(self, 'zarr_object', timspeak.io_interface.output.out_formats.zarr_object.ZARRObject)

    @staticmethod
    def read_mem_map(
         *,
         file_name: str,
         mmap_name: str
    ) -> np.ndarray:
        return timspeak.io_interface.output.out_formats.zarr_object.ZARRObject.read_nparray(
            file_name=file_name,
            mmap_name=mmap_name
        )

    def record_sample(
        self,
        sample_file_name: str,
    ) -> None:
        from timspeak.platform_utilities.platform_settings import Platform
        current_platform = Platform()
        separator = current_platform.separator
        sample_name = sample_file_name.split(separator)[-1].split('.')[0]
        self.zarr_object.set_new_attribute(self.output_file_name, '/', 'sample_name', sample_name)

    def record_package_info(self) -> None:
        from timspeak import __version__
        self.zarr_object.set_new_attribute(self.output_file_name, '/', 'version', __version__)

    def record_acquisition(
         self,
         dia_data: np.ndarray,
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/acquisition/')
        self.zarr_object.set_new_nparray(self.output_file_name, '/acquisition/cycle', dia_data.cycle)
        self.zarr_object.set_new_nparray(self.output_file_name, '/acquisition/tof_indptr', dia_data.tof_indptr)

    def print_smoothing_data(
        self,
        smoothing_parameters: dict,
        smooth_intensity_values: np.ndarray
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/smoothing/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'algorithm_name', smoothing_parameters['algorithm_name'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'ppm_tolerance', smoothing_parameters['ppm_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'im_tolerance', smoothing_parameters['im_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'rt_tolerance', smoothing_parameters['rt_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'ppm_tolerance', smoothing_parameters['ppm_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/smoothing/', 'ppm_tolerance', smoothing_parameters['ppm_tolerance'])
        self.zarr_object.set_new_nparray(self.output_file_name, '/smoothing/smooth_intensity_values', smooth_intensity_values)

    def print_clustering_as_dataframe(
            self,
            cluster_parameters: dict,
            clusters3d_stats: np.ndarray
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/clustering/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/clustering/', 'algorithm_name', cluster_parameters['algorithm_name'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/clustering/', 'ppm_tolerance', cluster_parameters['ppm_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/clustering/', 'im_tolerance', cluster_parameters['im_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/clustering/', 'rt_tolerance', cluster_parameters['rt_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/clustering/', 'clustering_threshold', cluster_parameters['clustering_threshold'])
        self.zarr_object.set_new_group(self.output_file_name, '/clustering/as_dataframe/')
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/apex_pointer', clusters3d_stats.apex_indices)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/number_of_ions', clusters3d_stats.sizes)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/frame_group', clusters3d_stats.frame_groups)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/mz_weighted_average', clusters3d_stats.mz_values)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/im_weighted_average', clusters3d_stats.im_values)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/rt_weighted_average', clusters3d_stats.rt_values)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/as_dataframe/summed_intensity', clusters3d_stats.intensity_values)

    def print_clustering_raw_pointers(
            self,
            index_3d: np.ndarray,
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/clustering/raw_pointers/')
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/raw_pointers/indptr', index_3d.indptr)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/raw_pointers/indices', index_3d.values)

    def print_clustering_rt_projection(
        self,
        clusters3d_stats: np.ndarray,
        xics: np.ndarray,
        xic_indptr: np.ndarray,
        cycle_length: int
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/clustering/rt_projection/')
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/rt_projection/indptr', xic_indptr)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/rt_projection/summed_intensity_values', xics)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/rt_projection/start_index', clusters3d_stats.rt_lower_boundaries // cycle_length)

    def print_clustering_im_projection(
            self,
            clusters3d_stats: np.ndarray,
            mobilograms: np.ndarray,
            mobilogram_indptr: np.ndarray
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/clustering/im_projection/')
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/im_projection/indptr', mobilogram_indptr)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/im_projection/summed_intensity_values', mobilograms)
        self.zarr_object.set_new_nparray(self.output_file_name, '/clustering/im_projection/start_index', clusters3d_stats.im_lower_boundaries)

    def print_ms1_precursors(
            self,
            precursor_indices: np.ndarray,
            min_size: int
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/precursors/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/precursors/', 'min_size', min_size)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/precursors/cluster_pointers', precursor_indices)

    def print_ms1_isotopes_charge_2(
            self,
            isotopes_charge_2_parameters: dict,
            lower_isotope_pointers_2: np.ndarray,
            upper_isotope_pointers_2: np.ndarray,
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/isotopes/charge_2/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_2/', 'ppm_tolerance', isotopes_charge_2_parameters['ppm_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_2/', 'im_tolerance', isotopes_charge_2_parameters['im_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_2/', 'rt_tolerance', isotopes_charge_2_parameters['rt_tolerance'])
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_2/lower_isotope_pointers', lower_isotope_pointers_2)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_2/upper_isotope_pointers', upper_isotope_pointers_2)


    def print_ms1_isotopes_charge_2_metrics(
            self,
            ks_values_rt_2: np.ndarray,
            ks_values_im_2: np.ndarray
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/isotopes/charge_2/metrics')
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_2/metrics/ks_distance_im', ks_values_im_2)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_2/metrics/ks_distance_rt', ks_values_rt_2)

    def print_ms1_isotopes_charge_3(
            self,
            isotopes_charge_3_parameters: dict,
            lower_isotope_pointers_3: np.ndarray,
            upper_isotope_pointers_3: np.ndarray,
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/isotopes/charge_3/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_3/', 'ppm_tolerance', isotopes_charge_3_parameters['ppm_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_3/', 'im_tolerance', isotopes_charge_3_parameters['im_tolerance'])
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/isotopes/charge_3/', 'rt_tolerance', isotopes_charge_3_parameters['rt_tolerance'])
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_3/lower_isotope_pointers', lower_isotope_pointers_3)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_3/upper_isotope_pointers', upper_isotope_pointers_3)

    def print_ms1_isotopes_charge_3_metrics(
            self,
            ks_values_rt_3: np.ndarray,
            ks_values_im_3: np.ndarray
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/isotopes/charge_3/metrics')
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_3/metrics/ks_distance_im', ks_values_im_3)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_3/metrics/ks_distance_rt', ks_values_rt_3)

    def print_ms1_isotopes_charge_2_metrics_distance_im_rt(
            self,
            ks_values_2: np.ndarray
    ) -> None:
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_2/metrics/ks_distance_im_rt', ks_values_2)

    def print_ms1_isotopes_charge_3_metrics_distance_im_rt(
            self,
            ks_values_3: np.ndarray
    ) -> None:
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/isotopes/charge_3/metrics/ks_distance_im_rt', ks_values_3)

    def print_ms1_mono_isotopic_precursors(
            self,
            ks_2d_threshold: float,
            monoisotopic_precursors: np.ndarray,
            monoisotopic_charges: np.ndarray,
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms1/monoisotopic_precursors/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms1/monoisotopic_precursors/', 'ks_2d_threshold', ks_2d_threshold)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/monoisotopic_precursors/as_dataframe/precursor_pointers', monoisotopic_precursors)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms1/monoisotopic_precursors/as_dataframe/charge', monoisotopic_charges)

    def print_ms2_fragments(
        self,
        fragment_indices: np.ndarray,
        ms2_fragments_min_size: int
    ) -> None:
        self.zarr_object.set_new_group(self.output_file_name, '/ms2/fragments/')
        self.zarr_object.set_new_attribute(self.output_file_name, '/ms2/fragments/', 'min_size', ms2_fragments_min_size)
        self.zarr_object.set_new_nparray(self.output_file_name, '/ms2/fragments/cluster_pointers', fragment_indices)
