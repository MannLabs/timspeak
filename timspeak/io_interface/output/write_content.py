
import dataclasses

def hdf_function():
    import timspeak.io_interface.output.out_formats.hdf
    return timspeak.io_interface.output.out_formats.hdf.HDFFormat

def zarr_function():
    import timspeak.io_interface.output.out_formats.zarr
    return timspeak.io_interface.output.out_formats.zarr.ZARRFormat

def output_function(output_format: str):
    output_formats = {
        'hdf': hdf_function(),
        'zarr': zarr_function(),
    }
    return output_formats[output_format]

@dataclasses.dataclass(frozen=True)
class WriteObject:

    @staticmethod
    def init_writing_object(
            output_file_name: str,
    ) -> None:
        file_extension = output_file_name.split('.')[-1]
        return output_function(file_extension)(output_file_name)
