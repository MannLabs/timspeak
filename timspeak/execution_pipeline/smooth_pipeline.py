
import numpy as np
import timspeak.execution_pipeline.io_pipeline
import timspeak.peak_picker_algorithms.algorithm_selection

class SmoothPipeline(
	timspeak.execution_pipeline.io_pipeline.IOPipeline
):

	def smoothing(self) -> None:
		self.logger.root_logger.info('---------- SMOOTHING ----------')
		smooth_intensity_values = self.smooth_data()
		self.save_smoothed_values(smooth_intensity_values)
		self.create_mmaps_for_smoothing()

	def smooth_data(self) -> np.ndarray:
		self.smoothing_parameters = self.config_file_content['smoothing']
		smoother = timspeak.peak_picker_algorithms.algorithm_selection.smooth_algorithm(
			self.smoothing_parameters['algorithm_name'])(
			dia_data=self.dia_data,
			im_sigma=self.smoothing_parameters['im_sigma'],
			im_tolerance=self.smoothing_parameters['im_tolerance'],
			ppm_tolerance=self.smoothing_parameters['ppm_tolerance'],
			rt_sigma=self.smoothing_parameters['rt_sigma'],
			rt_tolerance=self.smoothing_parameters['rt_tolerance'],
		)
		smooth_intensity_values = smoother.smooth_all_scans()
		self.logger.root_logger.info('data smoothed')
		return smooth_intensity_values

	def save_smoothed_values(self, smooth_intensity_values: np.ndarray) -> None:
		self.output_format_object.print_smoothing_data(self.smoothing_parameters, smooth_intensity_values)
		self.logger.root_logger.info('smooth_intensity_values saved to file')

	def create_mmaps_for_smoothing(self):
		self.smooth_intensity_values = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/smoothing/smooth_intensity_values'
		)
		self.logger.root_logger.info('smooth_intensity_values mapped')
