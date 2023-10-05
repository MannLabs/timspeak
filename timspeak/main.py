
def main(
	input_file_name: str,
	sample_file_name: str = None,
	output_file_name: str = None
) -> None:
	import timspeak.execution_pipeline.main_pipeline
	object_execute_pipeline = timspeak.execution_pipeline.main_pipeline.MainPipeline(input_file_name, sample_file_name, output_file_name)
	object_execute_pipeline.run()

if __name__ == '__main__':
	import os
	input_file_name = os.path.join(os.getcwd(), 'timspeak/configuration_files/default_configuration.json')
	sample_file_name = None
	output_file_name = None
	sample_file_name = os.path.join(os.getcwd(), '20220923_TIMS03_PaSk_SA_HeLa_Evo05_21min_IM0713_classical_SyS_4MS_woCE_S6-B3_1_32404.d')
	output_file_name = os.path.join(os.getcwd(), 'cmdln_output.hdf')
	main(input_file_name, sample_file_name, output_file_name)
