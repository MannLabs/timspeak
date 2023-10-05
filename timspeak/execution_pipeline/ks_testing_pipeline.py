
import numpy as np
import timspeak.execution_pipeline.metrics_1dprojections_pipeline
import timspeak.peak_picker_algorithms.cluster.clusters_stats
import timspeak.statistical_utilities.ks_1d
import timspeak.statistical_utilities.ks_algorithms

class KsTestingPipeline(
	timspeak.execution_pipeline.metrics_1dprojections_pipeline.Metrics1dProjectionsPipeline
):

	def ks_testing(
		self,
		cluster_3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.logger.root_logger.info('---------- KS TESTING ----------')
		ks_tester = self.get_ks_tester(cluster_3d_stats)
		ks_values_rt_im2 = self.get_ks_values(ks_tester, self.isotopic_pairs_2, '2')
		self.save_ks_values_2(ks_values_rt_im2)
		self.create_mmaps_for_ks_values_2()
		ks_values_rt_im3 = self.get_ks_values(ks_tester, self.isotopic_pairs_3, '3')
		self.save_ks_values_3(ks_values_rt_im3)
		self.create_mmaps_for_ks_values_3()

	def get_ks_tester(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> timspeak.statistical_utilities.ks_algorithms.KolmogorovSmirnovTester:
		ks_tester = timspeak.statistical_utilities.ks_algorithms.KolmogorovSmirnovTester(
			dia_data=self.dia_data,
			index3d=self.cluster_index,
			precursor_index=self.precursor_index,
			cluster3d_stats=cluster3d_stats,
			expanded_index_pointers=self.expanded_index_pointers,
		)
		self.logger.root_logger.info('ks_tester generated')
		return ks_tester

	def get_ks_values(
		self,
		ks_tester: timspeak.statistical_utilities.ks_algorithms.KolmogorovSmirnovTester,
		isotopic_pairs: np.ndarray,
		charge_str: str
	) -> np.ndarray:
		(
			lower_clusters,
			upper_clusters,
			ks_values,
			p_values,
		) = ks_tester.between_all_clusters(isotopic_pairs)
		np.bincount(p_values > 0.01)
		self.logger.root_logger.info(f'ks_values_{charge_str} generated')
		return ks_values

	def save_ks_values_2(
		self,
		ks_values_2: np.ndarray
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_2_metrics_distance_im_rt(
			ks_values_2
		)
		self.logger.root_logger.info('ks_values_2 saved')

	def create_mmaps_for_ks_values_2(self) -> None:
		self.ks_values_rt_im_2 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_2/metrics/ks_distance_im_rt'
		)
		self.logger.root_logger.info('ks_values_rt_im_2 mapped')

	def save_ks_values_3(
		self,
		ks_values_3: np.ndarray
	) -> None:
		self.output_format_object.print_ms1_isotopes_charge_3_metrics_distance_im_rt(
			ks_values_3
		)
		self.logger.root_logger.info('ks_values_3 saved')

	def create_mmaps_for_ks_values_3(self) -> None:
		self.ks_values_rt_im_3 = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/isotopes/charge_3/metrics/ks_distance_im_rt'
		)
		self.logger.root_logger.info('ks_values_rt_im_3 mapped')
