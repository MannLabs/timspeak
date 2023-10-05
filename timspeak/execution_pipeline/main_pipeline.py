
import numpy as np
import timspeak.execution_pipeline.ms2_fragments_pipeline
import timspeak.statistical_utilities.ks_1d

class MainPipeline(
    timspeak.execution_pipeline.ms2_fragments_pipeline.Ms2FragmentsPipeline
):

    def __init__(
        self,
        config_file_name: str,
        cmdln_sample_file_name: str = None,
        cmdln_output_file_name: str = None
    ) -> None:
        object.__setattr__(self, 'config_file_name', config_file_name)
        if cmdln_sample_file_name is not None:
            object.__setattr__(self, 'cmdln_sample_file_name', cmdln_sample_file_name)
        if cmdln_output_file_name is not None:
            object.__setattr__(self, 'cmdln_output_file_name', cmdln_output_file_name)

    def run(self) -> None:
        self.initialize_logger()
        self.set_input_objects()
        self.set_number_of_threads()
        self.set_output_objects()
        self.save_sample_info()
        self.save_package_info()
        self.load_dia_data()
        self.get_cycle_lenght()
        self.save_acquisition()
        self.smoothing()
        cluster3d_stats = self.clustering()
        self.ms1_precursors(cluster3d_stats)
        self.deisotoping(cluster3d_stats)
        self.metrics_for_1d_projections()
        self.ks_testing(cluster3d_stats)
        self.mono_isotopes()
        self.ms2_fragments(cluster3d_stats)
        self.logger.root_logger.info('execution ended')
