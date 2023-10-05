
import dataclasses
import abc
import numpy as np

@dataclasses.dataclass(frozen=False)
class IClusterData(abc.ABC):

	@staticmethod
	@abc.abstractmethod
	def cluster_all_scans(self) -> np.ndarray:
		pass
