

import dataclasses
import timspeak.io_interface.configuration_templates


@dataclasses.dataclass(frozen=True)
class JSONFormat(
    timspeak.io_interface.configuration_templates.IConfigFileContent
):

    @staticmethod
    def read_file_content(config_file_name: str) -> dict:
        import json
        file_content = None
        with open(config_file_name) as file_handler:
            file_content = json.load(file_handler)
        return file_content

