

import dataclasses


@dataclasses.dataclass(frozen=True)
class ConfigFileExtensions:

    @staticmethod
    def get_config_file_extensions() -> dict:
        from os import listdir
        from os.path import dirname
        from timspeak.platform_utilities.platform_settings import Platform
        current_platform = Platform()
        platform_separator = current_platform.separator
        formats_directory = dirname(__file__) + f'{platform_separator}in_formats'
        return {file.split('.')[0]: 1 for file in listdir(formats_directory) if file[0]!='_'}


@dataclasses.dataclass(frozen=True)
class ExtractConfigExtensions:

    @staticmethod
    def start_extracting() -> dict:
        return ConfigFileExtensions.get_config_file_extensions()

