

import dataclasses
import numpy as np
import timspeak.io_interface.configuration_templates
import timspeak.io_interface.output.out_formats.hdf_object


@dataclasses.dataclass(frozen=True)
class HDFFormat(
    timspeak.io_interface.configuration_templates.IOutputFileContent
):

    def __init__(
         self,
         output_file_name: str
    ) -> None:
        object.__setattr__(self, 'output_file_name', output_file_name)
        object.__setattr__(self, 'hdf_object', timspeak.io_interface.output.out_formats.hdf_object.HDFObject.from_file(output_file_name, new=True))

    @staticmethod
    def read_mem_map(*, file_name: str, mmap_name: str) -> np.ndarray:
        return timspeak.io_interface.output.out_formats.hdf_object.read_mmap(
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
        self.hdf_object.set_attr('sample_name', sample_name)

    def record_package_info(self) -> None:
        from timspeak import __version__
        self.hdf_object.set_attr('version', __version__)

    def record_acquisition(
         self,
         dia_data: np.ndarray,
    ) -> None:
        group_name = 'acquisition/'
        group = self.hdf_object.set_group(group_name)
        self.hdf_object.set_mmap(f'{group_name}cycle', dia_data.cycle)
        self.hdf_object.set_mmap(f'{group_name}tof_indptr', dia_data.tof_indptr)

    def print_smoothing_data(
        self,
        smoothing_parameters: dict,
        smooth_intensity_values: np.ndarray
    ) -> None:
        group_name = 'smoothing/'
        group = self.hdf_object.set_group(group_name)
        self.hdf_object.set_mmap(f'{group_name}smooth_intensity_values', smooth_intensity_values)
        group.set_attr('algorithm_name', smoothing_parameters['algorithm_name'])
        group.set_attr('ppm_tolerance', smoothing_parameters['ppm_tolerance'])
        group.set_attr('im_tolerance', smoothing_parameters['im_tolerance'])
        group.set_attr('rt_tolerance', smoothing_parameters['rt_tolerance'])
        group.set_attr('im_sigma', smoothing_parameters['im_sigma'])
        group.set_attr('rt_sigma', smoothing_parameters['rt_sigma'])

    def print_clustering_as_dataframe(
            self,
            cluster_parameters: dict,
            clusters3d_stats: np.ndarray
    ) -> None:
        group_name = 'clustering/'
        group = self.hdf_object.set_group(group_name)
        group.set_attr('algorithm_name', cluster_parameters['algorithm_name'])
        group.set_attr('ppm_tolerance', cluster_parameters['ppm_tolerance'])
        group.set_attr('rt_tolerance', cluster_parameters['rt_tolerance'])
        group.set_attr('im_tolerance', cluster_parameters['im_tolerance'])
        group.set_attr('clustering_threshold', cluster_parameters['clustering_threshold'])
        subgroup_name = f'{group_name}as_dataframe/'
        self.hdf_object.set_mmap(f'{subgroup_name}apex_pointer', clusters3d_stats.apex_indices)
        self.hdf_object.set_mmap(f'{subgroup_name}number_of_ions', clusters3d_stats.sizes)
        self.hdf_object.set_mmap(f'{subgroup_name}frame_group', clusters3d_stats.frame_groups)
        self.hdf_object.set_mmap(f'{subgroup_name}mz_weighted_average', clusters3d_stats.mz_values)
        self.hdf_object.set_mmap(f'{subgroup_name}im_weighted_average', clusters3d_stats.im_values)
        self.hdf_object.set_mmap(f'{subgroup_name}rt_weighted_average', clusters3d_stats.rt_values)
        self.hdf_object.set_mmap(f'{subgroup_name}summed_intensity', clusters3d_stats.intensity_values)

    def print_clustering_raw_pointers(
            self,
            index_3d: np.ndarray,
    ) -> None:
        subgroup_name = f'clustering/raw_pointers/'
        self.hdf_object.set_mmap(f'{subgroup_name}indptr', index_3d.indptr)
        self.hdf_object.set_mmap(f'{subgroup_name}indices', index_3d.values)

    def print_clustering_rt_projection(
        self,
        clusters3d_stats: np.ndarray,
        xics: np.ndarray,
        xic_indptr: np.ndarray,
        cycle_length: int
    ) -> None:
        subgroup_name = f'clustering/rt_projection/'
        self.hdf_object.set_mmap(f'{subgroup_name}indptr', xic_indptr)
        self.hdf_object.set_mmap(f'{subgroup_name}summed_intensity_values', xics)
        self.hdf_object.set_mmap(f'{subgroup_name}start_index', clusters3d_stats.rt_lower_boundaries // cycle_length)

    def print_clustering_im_projection(
            self,
            clusters3d_stats: np.ndarray,
            mobilograms: np.ndarray,
            mobilogram_indptr: np.ndarray
    ) -> None:
        subgroup_name = f'clustering/im_projection/'
        self.hdf_object.set_mmap(f'{subgroup_name}indptr', mobilogram_indptr)
        self.hdf_object.set_mmap(f'{subgroup_name}summed_intensity_values', mobilograms)
        self.hdf_object.set_mmap(f'{subgroup_name}start_index', clusters3d_stats.im_lower_boundaries)

    def print_ms1_precursors(
            self,
            precursor_indices: np.ndarray,
            min_size: int
    ) -> None:
        group_name = 'ms1/precursors/'
        group = self.hdf_object.set_group(group_name)
        self.hdf_object.set_mmap(f'{group_name}cluster_pointers', precursor_indices)
        group.set_attr('min_size', min_size)

    def print_ms1_isotopes_charge_2(
            self,
            isotopes_charge_2_parameters: dict,
            lower_isotope_pointers_2: np.ndarray,
            upper_isotope_pointers_2: np.ndarray,
    ) -> None:
        group_name = 'ms1/isotopes/charge_2/'
        group = self.hdf_object.set_group(group_name)
        group.set_attr('im_tolerance', isotopes_charge_2_parameters['im_tolerance'])
        group.set_attr('ppm_tolerance', isotopes_charge_2_parameters['ppm_tolerance'])
        group.set_attr('rt_tolerance', isotopes_charge_2_parameters['rt_tolerance'])
        self.hdf_object.set_mmap(f'{group_name}lower_isotope_pointers', lower_isotope_pointers_2)
        self.hdf_object.set_mmap(f'{group_name}upper_isotope_pointers', upper_isotope_pointers_2)

    def print_ms1_isotopes_charge_2_metrics(
            self,
            ks_values_rt_2: np.ndarray,
            ks_values_im_2: np.ndarray
    ) -> None:
        group_name = 'ms1/isotopes/charge_2/metrics/'
        group = self.hdf_object.set_group(group_name)
        self.hdf_object.set_mmap(f'{group_name}ks_distance_im', ks_values_im_2)
        self.hdf_object.set_mmap(f'{group_name}ks_distance_rt', ks_values_rt_2)

    def print_ms1_isotopes_charge_3(
            self,
            isotopes_charge_3_parameters: dict,
            lower_isotope_pointers_3: np.ndarray,
            upper_isotope_pointers_3: np.ndarray,
    ) -> None:
        group_name = 'ms1/isotopes/charge_3/'
        group = self.hdf_object.set_group(group_name)
        group.set_attr('im_tolerance', isotopes_charge_3_parameters['im_tolerance'])
        group.set_attr('ppm_tolerance', isotopes_charge_3_parameters['ppm_tolerance'])
        group.set_attr('rt_tolerance', isotopes_charge_3_parameters['rt_tolerance'])
        self.hdf_object.set_mmap(f'{group_name}lower_isotope_pointers', lower_isotope_pointers_3)
        self.hdf_object.set_mmap(f'{group_name}upper_isotope_pointers', upper_isotope_pointers_3)

    def print_ms1_isotopes_charge_3_metrics(
            self,
            ks_values_rt_3: np.ndarray,
            ks_values_im_3: np.ndarray
    ) -> None:
        group_name = 'ms1/isotopes/charge_3/metrics/'
        group = self.hdf_object.set_group(group_name)
        self.hdf_object.set_mmap(f'{group_name}ks_distance_im', ks_values_im_3)
        self.hdf_object.set_mmap(f'{group_name}ks_distance_rt', ks_values_rt_3)

    def print_ms1_isotopes_charge_2_metrics_distance_im_rt(
        self,
        ks_values_2: np.ndarray
    ) -> None:
        group_name = 'ms1/isotopes/charge_2/metrics/'
        self.hdf_object.set_mmap(f'{group_name}ks_distance_im_rt', ks_values_2)

    def print_ms1_isotopes_charge_3_metrics_distance_im_rt(
            self,
            ks_values_3: np.ndarray
    ) -> None:
        group_name = 'ms1/isotopes/charge_3/metrics/'
        self.hdf_object.set_mmap(f'{group_name}ks_distance_im_rt', ks_values_3)

    def print_ms1_mono_isotopic_precursors(
            self,
            ks_2d_threshold: float,
            monoisotopic_precursors: np.ndarray,
            monoisotopic_charges: np.ndarray,
    ) -> None:
        group_name = 'ms1/monoisotopic_precursors/'
        group = self.hdf_object.set_group(group_name)
        group.set_attr('ks_2d_threshold', ks_2d_threshold)
        sub_group_name = f'{group_name}as_dataframe/'
        self.hdf_object.set_mmap(f'{sub_group_name}precursor_pointers', monoisotopic_precursors)
        self.hdf_object.set_mmap(f'{sub_group_name}charge', monoisotopic_charges)

    def print_ms2_fragments(
        self,
        fragment_indices: np.ndarray,
        ms2_fragments_min_size: int
    ) -> None:
        group_name = 'ms2/'
        group = self.hdf_object.set_group(group_name)
        sub_group_name = f'{group_name}fragments/'
        sub_group = self.hdf_object.set_group(sub_group_name)
        sub_group.set_attr('min_size', ms2_fragments_min_size)
        self.hdf_object.set_mmap(f'{sub_group_name}cluster_pointers', fragment_indices)
