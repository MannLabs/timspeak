

import dataclasses


@dataclasses.dataclass(frozen=True)
class OutputFileExtensions:

    @staticmethod
    def get_config_file_extensions() -> dict:
        from os import listdir
        from os.path import dirname
        from timspeak.platform_utilities.platform_settings import Platform
        current_platform = Platform()
        platform_separator = current_platform.separator
        formats_directory = dirname(__file__) + f'{platform_separator}out_formats'
        return {file.split('.')[0]: 1 for file in listdir(formats_directory) if '_' not in file}


@dataclasses.dataclass(frozen=True)
class ExtractOutputExtensions:

    @staticmethod
    def start_extracting() -> dict:
        return OutputFileExtensions.get_config_file_extensions()

