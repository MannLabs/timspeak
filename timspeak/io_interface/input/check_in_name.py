

import dataclasses
import timspeak.io_interface.configuration_templates

@dataclasses.dataclass(frozen=True)
class CheckInFileExtension(
    timspeak.io_interface.configuration_templates.IConfigFileName
):

    @staticmethod
    def check_file_name(
        config_file_name: str,
        config_registered_extensions: dict,
    ) -> str:
        file_extension = config_file_name.split('.')[-1]
        if file_extension in config_registered_extensions.keys():
            return CheckInFileExists.check_file_name(config_file_name, config_registered_extensions)
        else:
            raise NameError(f'Configuration file {config_file_name} has not a valid extension.')


class CheckInFileExists(
    timspeak.io_interface.configuration_templates.IConfigFileName
):

    @staticmethod
    def check_file_name(
        config_file_name: str,
        config_registered_extensions: dict,
    ) -> str:
        from os.path import isfile
        if isfile(config_file_name):
            return config_file_name
        else:
            raise NameError(f'Configuration file {config_file_name} was not found.')


class CheckInName:

    @staticmethod
    def start_checking(
        config_file_name: str,
        config_registered_extensions: dict
    ) -> str:
        return CheckInFileExtension.check_file_name(config_file_name, config_registered_extensions)

