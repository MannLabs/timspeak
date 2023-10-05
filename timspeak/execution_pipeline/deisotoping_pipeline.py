
import numpy as np
import timspeak.execution_pipeline.ms1_precursors_pipeline
import timspeak.peak_picker_algorithms.algorithm_selection
import timspeak.data_handlers.indexing
import timspeak.peak_picker_algorithms.cluster.clusters_stats
import timspeak.peak_picker_algorithms.isotope.deisotoping


class DeisotopingPipeline(
	timspeak.execution_pipeline.ms1_precursors_pipeline.MS1PrecursorsPipeline
):
	def deisotoping(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.logger.root_logger.info('---------- DEISOTOPING ----------')
		lower_isotope_pointers_2, upper_isotope_pointers_2 = self.get_lower_upper_isotope_pointers_2(cluster3d_stats)
		self.save_isotope_pointers_2(lower_isotope_pointers_2, upper_isotope_pointers_2)
		self.create_mmaps_for_isotope_pointers_2()
		lower_isotope_pointers_3, upper_isotope_pointers_3 = self.get_lower_upper_isotope_pointers_3(cluster3d_stats)
		self.save_isotope_pointers_3(lower_isotope_pointers_3, upper_isotope_pointers_3)
		self.create_mmaps_for_isotope_pointers_3()

	def get_lower_upper_isotope_pointers_2(
		self,
		cluster3d_stats
	) -> tuple:
		self.ms1_isotopes_charge_2_parameters = self.config_file_content['ms1']['isotopes']['charge_2']
		deisotoper_2 = timspeak.peak_picker_algorithms.isotope.deisotoping.ChargeDeisotoper(
			dia_data=self.dia_data,
			index=self.precursor_index,
			cluster_stats=cluster3d_stats,
			charge=2,
			im_tolerance=self.ms1_isotopes_charge_2_parameters['im_tolerance'],
			ppm_tolerance=self.ms1_isotopes_charge_2_parameters['ppm_tolerance'],
			rt_tolerance=self.ms1_isotopes_charge_2_parameters['rt_tolerance']
		)
		isotopic_pairs_2 = deisotoper_2.deisotope_all_scans()
		lower_isotope_pointers_2 = np.flatnonzero(isotopic_pairs_2 != -1)
		upper_isotope_pointers_2 = isotopic_pairs_2[lower_isotope_pointers_2]
		self.isotopic_pairs_2 = isotopic_pairs_2
		self.logger.root_logger.info('lower and upper_isotope_pointers_2 generated')
		return lower_isotope_pointers_2, upper_isotope_pointers_2

	def save_isotope_pointers_2(
		self,
		lower_isotope_pointers_2: np.ndarray,
		upper_isotope_pointers_2: np.ndarray,
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_2(
			self.ms1_isotopes_charge_2_parameters,
			lower_isotope_pointers_2,
			upper_isotope_pointers_2
		)
		self.logger.root_logger.info('isotope_pointers_2 saved')

	def create_mmaps_for_isotope_pointers_2(self) -> None:
		self.lower_isotope_pointers_2 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_2/lower_isotope_pointers'
		)
		self.upper_isotope_pointers_2 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_2/upper_isotope_pointers'
		)
		self.logger.root_logger.info('isotope_pointers_2 mapped')

	def get_lower_upper_isotope_pointers_3(
		self,
		cluster3d_stats
	) -> tuple:
		self.ms1_isotopes_charge_3_parameters = self.config_file_content['ms1']['isotopes']['charge_3']
		deisotoper_3 = timspeak.peak_picker_algorithms.isotope.deisotoping.ChargeDeisotoper(
			dia_data=self.dia_data,
			index=self.precursor_index,
			cluster_stats=cluster3d_stats,
			charge=3,
			im_tolerance=self.ms1_isotopes_charge_3_parameters['im_tolerance'],
			ppm_tolerance=self.ms1_isotopes_charge_3_parameters['ppm_tolerance'],
			rt_tolerance=self.ms1_isotopes_charge_3_parameters['rt_tolerance']
		)
		isotopic_pairs_3 = deisotoper_3.deisotope_all_scans()
		lower_isotope_pointers_3 = np.flatnonzero(isotopic_pairs_3 != -1)
		upper_isotope_pointers_3 = isotopic_pairs_3[lower_isotope_pointers_3]
		self.isotopic_pairs_3 = isotopic_pairs_3
		self.logger.root_logger.info('lower and upper_isotope_pointers_3 generated')
		return lower_isotope_pointers_3, upper_isotope_pointers_3

	def save_isotope_pointers_3(
		self,
		lower_isotope_pointers_3: np.ndarray,
		upper_isotope_pointers_3: np.ndarray,
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_3(
			self.ms1_isotopes_charge_3_parameters,
			lower_isotope_pointers_3,
			upper_isotope_pointers_3
		)
		self.logger.root_logger.info('isotope_pointers_3 saved')

	def create_mmaps_for_isotope_pointers_3(self) -> None:
		self.lower_isotope_pointers_3 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_3/lower_isotope_pointers'
		)
		self.upper_isotope_pointers_3 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_3/upper_isotope_pointers'
		)
		self.logger.root_logger.info('isotope_pointers_3 mapped')
