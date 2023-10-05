
import zarr
import numpy as np
import mmap

class ZARRObject:

    @staticmethod
    def set_new_attribute(
        file_name: str,
        group_name: str,
        attribute_key: str,
        attribute_val: any
    ) -> None:
        with zarr.open(file_name, mode='a') as file:
            if group_name in file:
                group = file[group_name]
            else:
                group = file.create_group(group_name)
            group.attrs[attribute_key] = attribute_val

    @staticmethod
    def set_new_group(
         file_name: str,
         group_name: str,
    ) -> None:
        with zarr.open(file_name, mode='a') as file:
            if group_name not in file:
                file.create_group(group_name)

    @staticmethod
    def set_new_nparray(
         file_name: str,
         nparray_key: str,
         nparray_val: np.array,
    ) -> None:
        with zarr.open(file_name, mode='a') as file:
            if nparray_key not in file:
                file.create_dataset(
                    nparray_key,
                    data=nparray_val,
                    shape=nparray_val.shape,
                    dtype=nparray_val.dtype,
                    chunks=(None,),
                    compressor=None
                )

    @staticmethod
    def read_nparray(
         file_name: str,
         mmap_name: str
    ) -> np.array:
        offset = 0
        shape = None
        dtype = None
        nchunks = None
        with zarr.open(file_name, mode='r') as file:
            array = file[mmap_name]
            shape = array.shape
            dtype = array.dtype
            nchunks = array.nchunks
        mapped_file = file_name + '/' + mmap_name + '/0'
        with open(mapped_file, "rb") as file:
            mmap_obj = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            m1 = np.frombuffer(
                mmap_obj,
                dtype=dtype,
                count=np.prod(shape),
                offset=offset
            ).reshape(shape)
            return m1
