
import dataclasses
import abc
import numpy as np

@dataclasses.dataclass(frozen=False)
class ISmoothData(abc.ABC):

	@staticmethod
	@abc.abstractmethod
	def smooth_all_scans(self) -> np.ndarray:
		pass
