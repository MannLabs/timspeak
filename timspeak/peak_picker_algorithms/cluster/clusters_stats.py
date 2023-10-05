
import pandas as pd
import numpy as np
import alphatims.dia_data
import dataclasses

import timspeak.data_handlers.indexing
import timspeak.statistical_utilities.stats
import timspeak.performance_utilities.compiling

@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class Clusters3D:

	def __init__(
		self,
		dia_data: alphatims.dia_data.DiaData,
		sparse_index: timspeak.data_handlers.indexing.SparseIndex,
		smooth_intensity_values: np.ndarray,
	):
		object.__setattr__(self, "dia_data", dia_data)
		object.__setattr__(self, "sparse_index", sparse_index)
		object.__setattr__(self, "smooth_intensity_values", smooth_intensity_values)
		self.__post_init__()

	def __len__(self):
		return len(self.mz_values)

	def __post_init__(self):
		expanded_index_pointers = np.repeat(
			np.arange(len(self.dia_data.tof_indptr) - 1),
			np.diff(self.dia_data.tof_indptr),
		)
		rt_boundaries = self.__calculate_rt_boundaries(
			expanded_index_pointers
		)
		im_boundaries = self.__calculate_im_boundaries(
			expanded_index_pointers
		)
		object.__setattr__(self,	"rt_values", self.__calculate_rt_values(expanded_index_pointers))
		object.__setattr__(self, "im_values", self.__calculate_im_values(expanded_index_pointers))
		object.__setattr__(self, "mz_values", self.__calculate_mz_values())
		object.__setattr__(self, "intensity_values", self.__calculate_intensity_values())
		object.__setattr__(self, "apex_indices", self.__calculate_apex_indices())
		object.__setattr__(self, "frame_groups", self.__calculate_frame_groups())
		object.__setattr__(self, "rt_lower_boundaries", rt_boundaries[:, 0])
		object.__setattr__(self, "rt_upper_boundaries", rt_boundaries[:, 1])
		object.__setattr__(self, "im_lower_boundaries", im_boundaries[:, 0])
		object.__setattr__(self, "im_upper_boundaries", im_boundaries[:, 1])
		object.__setattr__(self, "sizes", np.diff(self.sparse_index.indptr))

	def as_dataframe(self, indices=..., *, columns=None) -> pd.DataFrame:
		if columns is None:
			columns = self.columns
		return pd.DataFrame(
			{
				name: self.__getattribute__(name)[indices] for name in columns
			}
		)

	def __calculate_rt_values(
		self,
		expanded_index_pointers: np.ndarray
	) -> np.ndarray:
		rt_calculator3d = timspeak.statistical_utilities.stats.RtCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
			expanded_index_pointers=expanded_index_pointers,
		)
		rt_values = rt_calculator3d.calculate()
		return rt_values

	def __calculate_im_values(
		self,
		expanded_index_pointers: np.ndarray
	) -> np.ndarray:
		im_calculator3d = timspeak.statistical_utilities.stats.ImCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
			expanded_index_pointers=expanded_index_pointers,
		)
		im_values = im_calculator3d.calculate()
		return im_values

	def __calculate_mz_values(self) -> np.ndarray:
		mz_calculator3d = timspeak.statistical_utilities.stats.MzCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
		)
		mz_values = mz_calculator3d.calculate()
		return mz_values

	def __calculate_intensity_values(self) -> np.ndarray:
		intensity_calculator3d = timspeak.statistical_utilities.stats.IntensityCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
		)
		intensity_values = intensity_calculator3d.calculate()
		return intensity_values

	def __calculate_apex_indices(self) -> np.ndarray:
		apex_calculator3d = timspeak.statistical_utilities.stats.ApexCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
			smooth_intensity_values=self.smooth_intensity_values,
		)
		apex_indices = apex_calculator3d.calculate()
		return apex_indices

	def __calculate_im_boundaries(
		self,
		expanded_index_pointers: np.ndarray
	) -> np.ndarray:
		im_boundary_calculator3d = timspeak.statistical_utilities.stats.IMBoundaryCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
			expanded_index_pointers=expanded_index_pointers,
		)
		im_boundaries = im_boundary_calculator3d.calculate()
		return im_boundaries

	def __calculate_rt_boundaries(
		self,
		expanded_index_pointers: np.ndarray
	) -> np.ndarray:
		rt_boundary_calculator3d = timspeak.statistical_utilities.stats.RTBoundaryCalculator(
			dia_data=self.dia_data,
			index=self.sparse_index,
			expanded_index_pointers=expanded_index_pointers,
		)
		rt_boundaries = rt_boundary_calculator3d.calculate()
		return rt_boundaries

	def __calculate_frame_groups(self) -> np.ndarray:
		frames = np.searchsorted(
			self.dia_data.tof_indptr[::len(self.dia_data.im_values)],
			self.sparse_index.values[self.sparse_index.indptr[:-1]],
			"right"
		) - 1
		frame_groups = (frames - 1) % self.dia_data.cycle.shape[1]
		return frame_groups
