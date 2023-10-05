

import dataclasses
import timspeak.io_interface.configuration_templates


@dataclasses.dataclass(frozen=True)
class YAMLFormat(
    timspeak.io_interface.configuration_templates.IConfigFileContent
):

    @staticmethod
    def read_file_content(config_file_name: str) -> dict:
        import yaml
        file_content = None
        with open(config_file_name, 'r') as file_handler:
            file_content = yaml.safe_load(file_handler)
        return file_content

