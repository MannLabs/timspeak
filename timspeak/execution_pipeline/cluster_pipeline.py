
import numpy as np
import timspeak.execution_pipeline.smooth_pipeline
import timspeak.peak_picker_algorithms.algorithm_selection
import timspeak.data_handlers.indexing
import timspeak.peak_picker_algorithms.cluster.clusters_stats

class ClusterPipeline(
	timspeak.execution_pipeline.smooth_pipeline.SmoothPipeline
):

	def clustering(
		self
	) -> timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D:
		self.logger.root_logger.info('---------- CLUSTERING ----------')
		index_3d = self.cluster_data()
		self.save_index_3d_raw_pointers(index_3d)
		self.create_mmaps_for_clustering_raw_pointers(index_3d)
		clusters3d_stats = self.get_cluster_stats(index_3d)
		self.save_clusters3d_stats_as_dataframe(clusters3d_stats)
		self.get_expanded_index_pointers()
		xics, xic_indptr = self.get_clustering_rt_projections(index_3d, clusters3d_stats)
		self.save_clustering_rt_projections(clusters3d_stats, xics, xic_indptr)
		self.create_mmaps_for_clustering_rt_projections()
		mobilograms, mobilogram_indptr = self.get_clustering_im_projections(index_3d, clusters3d_stats)
		self.save_clustering_im_projections(clusters3d_stats, mobilograms, mobilogram_indptr)
		self.create_mmaps_for_clustering_im_projections()
		self.generate_sparse_indices()
		return clusters3d_stats

	def cluster_data(
		self
	) -> timspeak.data_handlers.indexing.SparseIndex:
		self.clustering_parameters = self.config_file_content['clustering']
		clusterer = timspeak.peak_picker_algorithms.algorithm_selection.cluster_algorithm(
			self.clustering_parameters['algorithm_name'])(
			dia_data=self.dia_data,
			smooth_intensity_values=self.smooth_intensity_values,
			ppm_tolerance=self.clustering_parameters['ppm_tolerance'],
			im_tolerance=self.clustering_parameters['im_tolerance'],
			rt_tolerance=self.clustering_parameters['rt_tolerance'],
			clustering_threshold=self.clustering_parameters['clustering_threshold']
		)
		index_3d = clusterer.cluster_all_scans()
		self.logger.root_logger.info('data clustered')
		return index_3d

	def save_index_3d_raw_pointers(
		self,
		index_3d: timspeak.data_handlers.indexing.SparseIndex
	) -> None:
		self.output_format_object.print_clustering_raw_pointers(index_3d)
		self.logger.root_logger.info('index_3d saved to file')

	def create_mmaps_for_clustering_raw_pointers(
		self,
		index_3d: timspeak.data_handlers.indexing.SparseIndex
	) -> None:
		self.cluster_indptr = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/raw_pointers/indptr'
		)
		self.cluster_indices = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/raw_pointers/indices'
		)
		self.logger.root_logger.info('cluster_indptr and cluster_indices mapped')

	def get_cluster_stats(
		self,
		index_3d: timspeak.data_handlers.indexing.SparseIndex
	) -> timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D:
		cluster3d_stats = timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D(
			dia_data=self.dia_data,
			sparse_index=index_3d,
			smooth_intensity_values=self.smooth_intensity_values
		)
		self.logger.root_logger.info('cluster3d_stats calculated')
		return cluster3d_stats

	def save_clusters3d_stats_as_dataframe(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.output_format_object.print_clustering_as_dataframe(
			self.clustering_parameters,
			cluster3d_stats
		)
		self.logger.root_logger.info('cluster3d_stats saved to file')

	def get_expanded_index_pointers(self) -> None:
		self.expanded_index_pointers = np.repeat(
			np.arange(len(self.dia_data.tof_indptr) - 1),
			np.diff(self.dia_data.tof_indptr),
		)
		self.logger.root_logger.info('expanded_index_pointers calculated')

	def get_clustering_rt_projections(
		self,
		index_3d: timspeak.data_handlers.indexing.SparseIndex,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> tuple:
		pc = timspeak.statistical_utilities.ks_algorithms.XICCreator(
			dia_data=self.dia_data,
			index3d=index_3d,
			cluster3d_stats=cluster3d_stats,
			expanded_index_pointers=self.expanded_index_pointers,
		)
		xics, xic_indptr = pc.create_xics()
		self.logger.root_logger.info('xics, xic_indptr calculated')
		return xics, xic_indptr

	def save_clustering_rt_projections(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D,
		xics: np.ndarray,
		xic_indptr: np.ndarray
	) -> None:
		self.output_format_object.print_clustering_rt_projection(
			cluster3d_stats,
			xics,
			xic_indptr,
			self.cycle_length
		)
		self.logger.root_logger.info('clustering rt_projections saved to file')

	def create_mmaps_for_clustering_rt_projections(self) -> None:
		self.xic_indptr = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/rt_projection/indptr'
		)
		self.xics = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/rt_projection/summed_intensity_values'
		)
		self.xic_offsets = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/rt_projection/start_index'
		)
		self.logger.root_logger.info('xic_indptr, xics and xic_offsets mapped')

	def get_clustering_im_projections(
		self,
		index_3d: timspeak.data_handlers.indexing.SparseIndex,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> tuple:
		mc = timspeak.statistical_utilities.ks_algorithms.MobilogramCreator(
			dia_data=self.dia_data,
			index3d=index_3d,
			cluster3d_stats=cluster3d_stats,
			expanded_index_pointers=self.expanded_index_pointers
		)
		mobilograms, mobilogram_indptr = mc.create_mobilograms()
		self.logger.root_logger.info('mobilograms, mobilogram_indptr calculated')
		return mobilograms, mobilogram_indptr

	def save_clustering_im_projections(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D,
		mobilograms: np.ndarray,
		mobilogram_indptr: np.ndarray
	) -> None:
		self.output_format_object.print_clustering_im_projection(
			cluster3d_stats,
			mobilograms,
			mobilogram_indptr,
		)
		self.logger.root_logger.info('clustering im_projections saved to file')

	def create_mmaps_for_clustering_im_projections(self) -> None:
		self.mobilogram_indptr = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/im_projection/indptr'
		)
		self.mobilograms = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/im_projection/summed_intensity_values'
		)
		self.mobilogram_offsets = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/clustering/im_projection/start_index'
		)
		self.logger.root_logger.info('xic_indptr, xics and xic_offsets mapped')

	def generate_sparse_indices(self) -> None:
		self.cluster_index = timspeak.data_handlers.indexing.SparseIndex(
			indptr=self.cluster_indptr,
			values=self.cluster_indices
		)
		self.xic_index = timspeak.data_handlers.indexing.SparseIndex(
			indptr=self.xic_indptr,
			values=self.xics
		)
		self.mobilogram_index = timspeak.data_handlers.indexing.SparseIndex(
			indptr=self.mobilogram_indptr,
			values=self.mobilograms
		)
		self.logger.root_logger.info('sparse cluster_index, xic_index and mobilogram_index generated')
