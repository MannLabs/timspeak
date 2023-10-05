
import numpy as np
import timspeak.execution_pipeline.monoisotopes_pipeline
import timspeak.statistical_utilities.ks_algorithms

class Ms2FragmentsPipeline(
	timspeak.execution_pipeline.monoisotopes_pipeline.MonoIsotopesPipeline
):

	def ms2_fragments(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.logger.root_logger.info('---------- MS2 FRAGMENTS ----------')
		self.get_fragments_indices(cluster3d_stats)
		self.get_fragments_indptr(cluster3d_stats)
		self.get_fragments_sparse_index()
		self.save_fragments_indices()

	def get_fragments_indices(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		self.ms2_fragments_min_size = int(self.config_file_content['ms2']['fragments']['min_size'])
		fragment_indices = np.flatnonzero(
			(cluster3d_stats.frame_groups > 0) &
			(cluster3d_stats.sizes >= self.ms2_fragments_min_size)
		)
		o = np.argsort(cluster3d_stats.apex_indices[fragment_indices])
		fragment_indices = fragment_indices[o]
		self.fragment_indices = fragment_indices
		self.logger.root_logger.info('ms2 fragment_indices generated')

	def get_fragments_indptr(
		self,
		cluster3d_stats: timspeak.peak_picker_algorithms.cluster.clusters_stats.Clusters3D
	) -> None:
		indptr_counts = np.searchsorted(
			self.dia_data.tof_indptr,
			cluster3d_stats.apex_indices[self.fragment_indices],
			"right",
		)
		fragment_indptr = np.bincount(indptr_counts, minlength=len(self.dia_data.tof_indptr))
		fragment_indptr = np.cumsum(fragment_indptr)
		self.fragment_indptr = fragment_indptr
		self.logger.root_logger.info('ms2 fragment_indptr generated')

	def get_fragments_sparse_index(self) -> None:
		self.fragment_index = timspeak.data_handlers.indexing.SparseIndex(
			indptr=self.fragment_indptr,
			values=self.fragment_indices,
		)
		self.logger.root_logger.info('ms2 sparse fragment_index generated')

	def save_fragments_indices(self) -> None:
		self.output_format_object.print_ms2_fragments(self.fragment_indices, self.ms2_fragments_min_size)
		self.logger.root_logger.info('MS2 fragment_indices saved')

