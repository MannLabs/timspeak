
import alphatims.dia_data
import timspeak.io_interface.logger
import timspeak.io_interface.input.extract_in_extensions
import timspeak.io_interface.input.check_in_name
import timspeak.io_interface.input.read_content
import timspeak.io_interface.output.extract_out_extensions
import timspeak.io_interface.output.check_out_name
import timspeak.io_interface.output.write_content

class IOPipeline:

	def initialize_logger(self) -> None:
		self.logger = timspeak.io_interface.logger.Logger()

	def set_input_objects(self) -> None:
		self.logger.root_logger.info('---------- SET INPUT OBJECTS ----------')
		self.check_config_file_name()
		self.read_config_file_content()

	def check_config_file_name(self) -> None:
		self.logger.root_logger.info(f'configuration file: {self.config_file_name}')
		object_input_extensions = timspeak.io_interface.input.extract_in_extensions.ExtractConfigExtensions()
		config_registered_extensions = object_input_extensions.start_extracting()
		ext_str = ', '.join(sorted(config_registered_extensions.keys()))
		self.logger.root_logger.info(f'valid configuration file extensions: {ext_str}')
		object_check_in_name = timspeak.io_interface.input.check_in_name.CheckInName()
		self.config_file_name_checked = object_check_in_name.start_checking(self.config_file_name,
															   config_registered_extensions)
		self.logger.root_logger.info(f'configuration file name checked')

	def read_config_file_content(self) -> None:
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		self.config_file_content = object_read_content.start_reading(self.config_file_name_checked)
		if hasattr(self, 'cmdln_sample_file_name'):
			self.config_file_content['sample_file_name'] = self.cmdln_sample_file_name
		if hasattr(self, 'cmdln_output_file_name'):
			self.config_file_content['output_file_name'] = self.cmdln_output_file_name
		self.logger.root_logger.info(f'sample name: {self.config_file_content["sample_file_name"]}')

	def set_number_of_threads(self) -> None:
		self.logger.root_logger.info('---------- SET NUMBER OF THREADS ----------')
		timspeak.performance_utilities.multiprocessing.set_threads(self.config_file_content['number_of_threads'])
		self.logger.root_logger.info(f'number of threads: {self.config_file_content["number_of_threads"]}')

	def set_output_objects(self) -> None:
		self.logger.root_logger.info('---------- SET OUTPUT OBJECTS ----------')
		self.check_output_file_name()
		self.initialize_output_object()

	def check_output_file_name(self) -> None:
		object_output_extensions = timspeak.io_interface.output.extract_out_extensions.ExtractOutputExtensions()
		output_registered_extensions = object_output_extensions.start_extracting()
		ext_str = ', '.join(sorted(output_registered_extensions.keys()))
		self.logger.root_logger.info(f'valid output file extensions: {ext_str}')
		object_check_out_name = timspeak.io_interface.output.check_out_name.CheckOutName()
		self.output_file_name = object_check_out_name.start_checking(self.config_file_content['output_file_name'],
														 output_registered_extensions)
		self.logger.root_logger.info(f'output file: {self.output_file_name}')

	def initialize_output_object(self) -> None:
		output_object = timspeak.io_interface.output.write_content.WriteObject()
		self.output_format_object = output_object.init_writing_object(self.output_file_name)

	def save_sample_info(self) -> None:
		self.output_format_object.record_sample(self.config_file_content['sample_file_name'])

	def save_package_info(self) -> None:
		self.output_format_object.record_package_info()

	def load_dia_data(self) -> None:
		self.logger.root_logger.info('---------- LOADING DATA ----------')
		self.dia_data = alphatims.dia_data.DiaData(
			dia_data=alphatims.bruker.TimsTOF(self.config_file_content['sample_file_name']))

	def get_cycle_lenght(self) -> None:
		self.cycle_length = self.dia_data.cycle.shape[1]

	def save_acquisition(self) -> None:
		self.output_format_object.record_acquisition(self.dia_data)
		self.logger.root_logger.info('acquisition (cycle and tof_indptr) saved to file')
