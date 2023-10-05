import dataclasses
import os

@dataclasses.dataclass(frozen=True)
class Platform:
	def __init__(self) -> None:
		import psutil
		import platform
		import timspeak
		object.__setattr__(self, 'platform', platform.platform())
		object.__setattr__(self, 'system', platform.system())
		object.__setattr__(self, 'release', platform.release())
		if platform.system() == 'Darwin':
			object.__setattr__(self, 'version', platform.mac_ver()[0])
		else:
			object.__setattr__(self, 'version', platform.version())
		object.__setattr__(self, 'machine', platform.machine())
		object.__setattr__(self, 'processor', platform.processor())
		object.__setattr__(self, 'python', platform.python_version())
		object.__setattr__(self, 'cpu_count', psutil.cpu_count())
		object.__setattr__(self, 'ram_available', f'{psutil.virtual_memory().available/(1024**3):.1f} GB')
		object.__setattr__(self, 'ram_total', f'{psutil.virtual_memory().total/(1024**3):.1f} GB')
		object.__setattr__(self, 'timspeak', timspeak.__version__)
		object.__setattr__(self, 'separator', self._get_platform_path_separator())

	def _get_platform_path_separator(self) -> str:
		return os.path.sep