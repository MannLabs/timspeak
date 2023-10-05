

import dataclasses


@dataclasses.dataclass(frozen=True)
class CheckOutFileExtension:

    @staticmethod
    def check_file_name(
            output_file_name: str,
            output_registered_extensions: dict,
    ) -> str:
        file_extension = output_file_name.split('.')[-1]
        if file_extension in output_registered_extensions.keys():
            return output_file_name
        else:
            raise NameError(f'Output file {output_file_name} has not a valid extension.')


class CheckOutName:

    @staticmethod
    def start_checking(
            output_file_name: str,
            output_registered_extensions: dict
    ) -> str:
        return CheckOutFileExtension.check_file_name(output_file_name, output_registered_extensions)

