
import numpy as np
import timspeak.execution_pipeline.cluster_pipeline
import timspeak.peak_picker_algorithms.cluster.clusters_stats

class MS1PrecursorsPipeline(
	timspeak.execution_pipeline.cluster_pipeline.ClusterPipeline
):

	def ms1_precursors(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.logger.root_logger.info('---------- MS1 PRECURSORS ----------')
		precursor_indices = self.get_precursor_indices(cluster3d_stats)
		self.save_precursor_indices(precursor_indices)
		self.create_mmaps_for_precursor_indices()
		self.generate_sparse_precursor_index(cluster3d_stats)

	def get_precursor_indices(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> np.ndarray:
		self.ms1_precursor_min_size = int(self.config_file_content['ms1']['precursors']['min_size'])
		precursor_indices = np.flatnonzero(
			(cluster3d_stats.frame_groups == 0) & (cluster3d_stats.sizes >= self.ms1_precursor_min_size))
		o = np.argsort(cluster3d_stats.apex_indices[precursor_indices])
		precursor_indices = precursor_indices[o]
		self.logger.root_logger.info('precursor_indices generated')
		return precursor_indices

	def save_precursor_indices(
		self,
		precursor_indices: np.ndarray
	) -> None:
		self.output_format_object.print_ms1_precursors(
			precursor_indices,
			self.ms1_precursor_min_size
		)
		self.logger.root_logger.info('precursor_indices saved')

	def create_mmaps_for_precursor_indices(self) -> None:
		self.precursor_indices = self.output_format_object.read_mem_map(
			file_name=self.output_file_name,
			mmap_name='/ms1/precursors/cluster_pointers'
		)
		self.logger.root_logger.info('precursor_indices mapped')

	def generate_sparse_precursor_index(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D,
	) -> None:
		indptr_counts = np.searchsorted(
			self.dia_data.tof_indptr,
			cluster3d_stats.apex_indices[self.precursor_indices],
			"right",
		)
		precursor_indptr = np.bincount(indptr_counts, minlength=len(self.dia_data.tof_indptr))
		precursor_indptr = np.cumsum(precursor_indptr)
		self.precursor_index = timspeak.data_handlers.indexing.SparseIndex(
			indptr=precursor_indptr,
			values=self.precursor_indices,
		)
		self.logger.root_logger.info('sparse precursor_index generated')
