
import numpy as np
import timspeak.execution_pipeline.deisotoping_pipeline
import timspeak.peak_picker_algorithms.cluster.clusters_stats
import timspeak.statistical_utilities.ks_1d

class Metrics1dProjectionsPipeline(
	timspeak.execution_pipeline.deisotoping_pipeline.DeisotopingPipeline
):

	def metrics_for_1d_projections(self):
		self.logger.root_logger.info('---------- METRICS 1D PROJECTIONS ----------')
		ks1_1d_xics = self.get_ks1_1d_xics()
		ks1_1d_im = self.get_ks1_1d_im()
		paired_indices_charge_2 = self.get_paired_indices(self.isotopic_pairs_2, '2')
		ks_values_rt_charge_2, ks_values_im_charge_2 = self.get_ks_values_rt_im_charge(ks1_1d_xics, ks1_1d_im,
																	    paired_indices_charge_2, '2')
		self.save_ks_values_rt_im_charge_2(ks_values_rt_charge_2, ks_values_im_charge_2)
		self.create_mmaps_for_ks_values_rt_im_charge_2()
		paired_indices_charge_3 = self.get_paired_indices(self.isotopic_pairs_3, '3')
		ks_values_rt_charge_3, ks_values_im_charge_3 = self.get_ks_values_rt_im_charge(ks1_1d_xics, ks1_1d_im,
																	    paired_indices_charge_3, '3')
		self.save_ks_values_rt_im_charge_3(ks_values_rt_charge_3, ks_values_im_charge_3)
		self.create_mmaps_for_ks_values_rt_im_charge_3()

	def get_ks1_1d_xics(
		self
	) -> timspeak.statistical_utilities.ks_1d.KSTester1D:
		cdf_with_offsets_xic = timspeak.statistical_utilities.ks_1d.CDFWithOffset(
			sparse_indices=self.xic_index,
			start_offsets=self.xic_offsets // self.cycle_length
		)
		ks1_1d_xics = timspeak.statistical_utilities.ks_1d.KSTester1D(
			cdf_with_offset=cdf_with_offsets_xic
		)
		self.logger.root_logger.info('ks1_1d_xics calculated')
		return ks1_1d_xics

	def get_ks1_1d_im(
		self
	) -> timspeak.statistical_utilities.ks_1d.KSTester1D:
		cdf_with_offsets_im = timspeak.statistical_utilities.ks_1d.CDFWithOffset(
			sparse_indices=self.mobilogram_index,
			start_offsets=self.mobilogram_offsets
		)
		ks1_1d_im = timspeak.statistical_utilities.ks_1d.KSTester1D(
			cdf_with_offset=cdf_with_offsets_im
		)
		self.logger.root_logger.info('ks1_1d_im calculated')
		return ks1_1d_im

	def get_paired_indices(
		self,
		isotopic_pairs: np.ndarray,
		charge_str: str
	) -> np.ndarray:
		selection_charge = np.flatnonzero(isotopic_pairs != -1)
		paired_indices_charge = np.array([selection_charge, isotopic_pairs[selection_charge]])
		paired_indices_charge = self.precursor_indices[paired_indices_charge.T]
		self.logger.root_logger.info(f'paired_indices_charge_{charge_str} calculated')
		return paired_indices_charge

	def get_ks_values_rt_im_charge(
		self,
		ks1_1d_xics: timspeak.statistical_utilities.ks_1d.KSTester1D,
		ks1_1d_im: timspeak.statistical_utilities.ks_1d.KSTester1D,
		paired_indices_charge: np.ndarray,
		charge_str: str
	) -> tuple:
		ks_values_rt_charge = ks1_1d_xics.calculate_all(paired_indices_charge)
		ks_values_im_charge = ks1_1d_im.calculate_all(paired_indices_charge)
		self.logger.root_logger.info(f'ks_values_rt_im_charge_{charge_str} calculated')
		return ks_values_rt_charge, ks_values_im_charge

	def save_ks_values_rt_im_charge_2(
		self,
		ks_values_rt_charge_2: np.ndarray,
		ks_values_im_charge_2: np.ndarray
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_2_metrics(
			ks_values_rt_charge_2,
			ks_values_im_charge_2
		)
		self.logger.root_logger.info('ks_values_rt_im_charge_2 saved')

	def create_mmaps_for_ks_values_rt_im_charge_2(
		self
	) -> None:
		self.ks_values_im_charge_2 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_2/metrics/ks_distance_im'
		)
		self.ks_values_rt_charge_2 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_2/metrics/ks_distance_rt'
		)
		self.logger.root_logger.info('ks_values_rt_im_charge_2 mapped')

	def save_ks_values_rt_im_charge_3(
		self,
		ks_values_rt_charge_3: np.ndarray,
		ks_values_im_charge_3: np.ndarray
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_3_metrics(
			ks_values_rt_charge_3,
			ks_values_im_charge_3
		)
		self.logger.root_logger.info('ks_values_rt_im_charge_3 saved')

	def create_mmaps_for_ks_values_rt_im_charge_3(
		self
	) -> None:
		self.ks_values_im_charge_3 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_3/metrics/ks_distance_im'
		)
		self.ks_values_rt_charge_3 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_3/metrics/ks_distance_rt'
		)
		self.logger.root_logger.info('ks_values_rt_im_charge_3 mapped')
