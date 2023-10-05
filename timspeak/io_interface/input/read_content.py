
import dataclasses

def json_function():
    import timspeak.io_interface.input.in_formats.json
    return timspeak.io_interface.input.in_formats.json.JSONFormat.read_file_content

def yaml_function():
    import timspeak.io_interface.input.in_formats.yaml
    return timspeak.io_interface.input.in_formats.yaml.YAMLFormat.read_file_content

def input_function(input_format: str):
    input_formats = {
        'json': json_function(),
        'yaml': yaml_function(),
    }
    return input_formats[input_format]

@dataclasses.dataclass(frozen=True)
class ReadContent:

    @staticmethod
    def start_reading(config_file_name: str) -> dict:
        file_extension = config_file_name.split('.')[-1]
        return input_function(file_extension)(config_file_name)
