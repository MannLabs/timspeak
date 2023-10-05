#!python


# builtin
import mmap
import os
import contextlib
import dataclasses

# external
import numpy as np
import h5py


def read_mmap(
    *,
    file_name: str,
    mmap_name: str,
    group_name: str = None,
) -> np.ndarray:
    if group_name is not None:
        mmap_name = f"{group_name}/{mmap_name}"
    with h5py.File(file_name, "r") as hdf_file:
        array = hdf_file[mmap_name]
        return _read_mmap(array, file_name)


def _read_mmap(array, file_name):
    offset = array.id.get_offset()
    shape = array.shape
    with open(file_name, "rb") as raw_hdf_file:
        mmap_obj = mmap.mmap(
            raw_hdf_file.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
        return np.frombuffer(
            mmap_obj,
            dtype=array.dtype,
            count=np.prod(shape),
            offset=offset
        ).reshape(shape)


def write_mmap(
    *,
    file_name: str,
    group_name: str,
    mmap_name: str,
    mmap_value: np.ndarray,
) -> np.ndarray:
    with get_or_create_group_from_hdf(file_name, group_name) as group:
        if mmap_name in group:
            del group[mmap_name]
        group[mmap_name] = mmap_value
    return read_mmap(
        file_name=file_name,
        mmap_name=mmap_name,
        group_name=group_name,
    )


def read_attr(
    *,
    file_name: str,
    group_name: str,
    attr_name: str,
) -> any:
    with h5py.File(file_name, "r") as hdf_file:
        group = hdf_file[group_name]
        return _read_attr(group, attr_name)


def _read_attr(group, attr_name):
    return group.attrs[attr_name]


def write_attr(
    *,
    file_name: str,
    group_name: str,
    attr_name: str,
    attr_value: any,
) -> any:
    with get_or_create_group_from_hdf(file_name, group_name) as group:
        group.attrs[attr_name] = attr_value
    return read_attr(
        file_name=file_name,
        group_name=group_name,
        attr_name=attr_name,
    )


@contextlib.contextmanager
def get_or_create_group_from_hdf(
    file_name: str,
    group_name: str
) -> h5py.Group:
    path_name = os.path.dirname(file_name)
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    with h5py.File(file_name, "a") as hdf_file:
        if group_name not in hdf_file:
            group = hdf_file.create_group(group_name)
        else:
            group = hdf_file[group_name]
        yield group


@dataclasses.dataclass(frozen=True)
class HDFObject:

    group_name: str

    def __init__(
        self,
        hdf_group: h5py.Group,
        file_name: str,
        group_name: str,
    ) -> None:
        object.__setattr__(self, "file_name", file_name)
        if not group_name.endswith("/"):
            group_name = f"{group_name}/"
        object.__setattr__(self, "group_name", group_name)
        self.set_attrs_from_group(hdf_group)
        self.set_arrays_from_group(hdf_group)

    def set_attrs_from_group(self, hdf_group):
        attrs = []
        for attr_name in hdf_group.attrs:
            object.__setattr__(self, attr_name, hdf_group.attrs[attr_name])
            attrs.append(attr_name)
        object.__setattr__(self, "attrs", attrs)

    def set_arrays_from_group(self, hdf_group):
        arrays = []
        for item_name in hdf_group:
            item = hdf_group[item_name]
            if isinstance(item, h5py.Dataset):
                mmap_array = _read_mmap(item, self.file_name)
                object.__setattr__(self, item_name, mmap_array)
                arrays.append(item_name)
            else:
                object.__setattr__(
                    self,
                    item_name,
                    type(self)(item, self.file_name, f"{self.group_name}{item_name}")
                )
        object.__setattr__(self, "arrays", arrays)

    @classmethod
    def from_file(cls, file_name: str, *, new: bool = False):
        file_name = os.path.abspath(file_name)
        path_name = os.path.dirname(file_name)
        if not os.path.exists(path_name):
            os.mkdir(path_name)
        mode = "w" if new else "r"
        with h5py.File(file_name, mode) as hdf_file:
            hdf_object =  cls(hdf_file, file_name, "")
            return hdf_object

    def set_attr(self, attr_name: str, attr_value: any) -> any:
        attr_value = write_attr(
            file_name=self.file_name,
            group_name=self.group_name,
            attr_name=attr_name,
            attr_value=attr_value,
        )
        object.__setattr__(self, attr_name, attr_value)
        if attr_name not in self.attrs:
            self.attrs.append(attr_name)
        return attr_value

    def set_mmap(self, mmap_name:str, mmap_value: np.ndarray) -> np.ndarray:
        mmap_value = write_mmap(
            file_name=self.file_name,
            group_name=self.group_name,
            mmap_name=mmap_name,
            mmap_value=mmap_value,
        )
        object.__setattr__(self, mmap_name, mmap_value)
        if mmap_name not in self.arrays:
            self.arrays.append(mmap_name)
        return mmap_value

    def set_group(self, group_name: str):
        full_group_name = f"{self.group_name}{group_name}"
        with get_or_create_group_from_hdf(
            self.file_name,
            full_group_name
        ) as group:
            group_object = type(self)(group, self.file_name, full_group_name)
            object.__setattr__(self, group_name, group_object)
        return group_object

