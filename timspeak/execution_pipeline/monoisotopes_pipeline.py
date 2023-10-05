
import numpy as np
import timspeak.execution_pipeline.ks_testing_pipeline
import timspeak.statistical_utilities.ks_algorithms

class MonoIsotopesPipeline(
	timspeak.execution_pipeline.ks_testing_pipeline.KsTestingPipeline
):

	def mono_isotopes(self) -> None:
		self.logger.root_logger.info('---------- DETERMINING MONO-ISOTOPES ----------')
		precursor_is_monoisotopic = self.get_monoisotopic_precursors()
		self.get_monoisotopic_charges(precursor_is_monoisotopic)
		self.save_monoisotopic_precursors_charges()

	def get_monoisotopic_precursors(self) -> np.ndarray:
		self.ks_2d_threshold = float(self.config_file_content['ms1']['isotopes']['monoisotopic_precursors']['ks_2d_threshold'])
		precursor_is_monoisotopic = np.zeros(len(self.precursor_indices), dtype=np.int64)
		precursor_is_monoisotopic[self.lower_isotope_pointers_2[self.ks_values_rt_im_2 < self.ks_2d_threshold]] += 2
		precursor_is_monoisotopic[self.lower_isotope_pointers_3[self.ks_values_rt_im_3 < self.ks_2d_threshold]] += 3
		precursor_is_monoisotopic[self.upper_isotope_pointers_2[self.ks_values_rt_im_2 < self.ks_2d_threshold]] = 0
		precursor_is_monoisotopic[self.upper_isotope_pointers_3[self.ks_values_rt_im_3 < self.ks_2d_threshold]] = 0
		self.monoisotopic_precursors = np.flatnonzero(
			(2 == precursor_is_monoisotopic) | (3 == precursor_is_monoisotopic))
		self.logger.root_logger.info('monoisotopic_precursors calculated')
		return precursor_is_monoisotopic

	def get_monoisotopic_charges(
		self,
		precursor_is_monoisotopic: np.ndarray
	) -> None:
		self.monoisotopic_charges = precursor_is_monoisotopic[self.monoisotopic_precursors]
		self.logger.root_logger.info('monoisotopic_charges calculated')

	def save_monoisotopic_precursors_charges(self):
		self.output_format_object.print_ms1_mono_isotopic_precursors(
			self.ks_2d_threshold,
			self.monoisotopic_precursors,
			self.monoisotopic_charges
		)
		self.logger.root_logger.info('monoisotopic_precursors and charges saved')
