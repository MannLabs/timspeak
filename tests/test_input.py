"""This module provides unit tests for timspeak input"""

import unittest


def get_timspeak_path() -> str:

	import os
	timspeak_path = os.path.dirname(os.getcwd())
	return timspeak_path


def add_timspeak_path(timspeak_path: str) -> None:
	import sys
	sys.path.append(timspeak_path)


def get_reference_content() -> dict:
	d_smoothing = {'algorithm_name': 'smoothing_algorithm_1',
				'im_sigma': 0.012,
				'im_tolerance': 0.004,
				'ppm_tolerance': 30.0,
				'rt_sigma': 1.5,
				'rt_tolerance': 1.5
				}
	d_clustering = {'algorithm_name': 'clustering_algorithm_1',
				 'im_tolerance': 0.004,
				 'ppm_tolerance': 20.0,
				 'rt_tolerance': 1.5,
				 'clustering_threshold': 5
				 }
	d_ms1 = {'precursors': {'min_size': 10},
		    'isotopes': {'charge_2': {'im_tolerance': 0.004,
								'ppm_tolerance': 20.0,
								'rt_tolerance': 1.5
								},
					  'charge_3': {'im_tolerance': 0.004,
								'ppm_tolerance': 20.0,
								'rt_tolerance': 1.5
								},
					  'monoisotopic_precursors': {'ks_2d_threshold': 0.4}
					  }
		    }
	d_ms2 = {'fragments': {'min_size': 5}}
	d = {}
	d['smoothing'] = d_smoothing
	d['clustering'] = d_clustering
	d['ms1'] = d_ms1
	d['ms2'] = d_ms2
	return d


class TestInput(unittest.TestCase):

	def test_load_json(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		self.assertTrue(config_file_content != None)

	def test_load_yaml(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.yaml')
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		self.assertTrue(config_file_content != None)

	def test_number_of_threads(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		self.assertTrue(config_file_content['number_of_threads'] > 0)

	def test_smoothing(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		reference = get_reference_content()
		self.assertTrue(config_file_content['smoothing'] == reference['smoothing'])

	def test_clustering(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		reference = get_reference_content()
		self.assertTrue(config_file_content['clustering'] == reference['clustering'])

	def test_ms1(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		reference = get_reference_content()
		self.assertTrue(config_file_content['ms1'] == reference['ms1'])

	def test_ms2(self) -> None:
		timspeak_path = get_timspeak_path()
		import os
		config_file_name = os.path.join(timspeak_path, 'default_configuration.json')
		add_timspeak_path(timspeak_path)
		import timspeak.io_interface.input.read_content
		config_file_content = None
		object_read_content = timspeak.io_interface.input.read_content.ReadContent()
		config_file_content = object_read_content.start_reading(config_file_name)
		reference = get_reference_content()
		self.assertTrue(config_file_content['ms2'] == reference['ms2'])
