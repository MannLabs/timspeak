import dataclasses

# external
import numpy as np
import alphatims.dia_data

# local
import timspeak.performance_utilities.compiling
import timspeak.performance_utilities.multiprocessing
import timspeak.data_handlers.sample_iterator
import timspeak.peak_picker_algorithms.smooth.smooth_templates


@timspeak.performance_utilities.compiling.njit_class
@dataclasses.dataclass(kw_only=True, frozen=False)
class Smoother(
    timspeak.peak_picker_algorithms.smooth.smooth_templates.ISmoothData
):
    """
    A class for smoothing DIA data using a specific algorithm.

    Parameters:
    - dia_data (alphatims.dia_data.DiaData): The DIA data to be smoothed.
    - frame_generator (timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator):
        An iterator for generating neighboring frames based on cyclic RT.
    - scan_generator (timspeak.data_handlers.sample_iterator.ImNeighborGenerator):
        An iterator for generating neighboring scans based on IM.
    - ion_pair_generator (timspeak.data_handlers.sample_iterator.MZIonPairGenerator):
        An iterator for generating ion pairs based on MZ.
    - im_sigma (float): The sigma value for IM correction. Default is None.
    - rt_sigma (float): The sigma value for RT correction. Default is None.
    - ppm_tolerance (float): The ppm tolerance for MZ correction. Default is 30.0.
    - im_tolerance (float): The IM tolerance for IM correction. Default is 0.01.
    - rt_tolerance (float): The RT tolerance for RT correction. Default is 3.0.
    - algorithm_name (str): The name of the smoothing algorithm. Default is 'smoothing_algorithm_1'.
    - algorithm_description (str): The description of the smoothing algorithm. Default is 'This is the original algorithm'.
    """

    dia_data: alphatims.dia_data.DiaData
    frame_generator: timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    scan_generator: timspeak.data_handlers.sample_iterator.ImNeighborGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    ion_pair_generator: timspeak.data_handlers.sample_iterator.MZIonPairGenerator = dataclasses.field(
        init=False,
        repr=False,
    )
    im_sigma: float = None # 0.01
    rt_sigma: float = None # 1.0
    ppm_tolerance: float = 30.0
    im_tolerance: float = 0.01
    rt_tolerance: float = 3.0

    algorithm_name: str = 'smoothing_algorithm_1'
    algorithm_description: str = 'This is the original algorithm'

    def __post_init__(self):
        if self.im_sigma is None:
            object.__setattr__(self, "im_sigma", self.im_tolerance / 3)
        if self.rt_sigma is None:
            object.__setattr__(self, "rt_sigma", self.rt_tolerance / 3)
        frame_generator = timspeak.data_handlers.sample_iterator.CyclicRtNeighborGenerator(
            dia_data = self.dia_data,
            rt_tolerance=self.rt_tolerance,
        )
        object.__setattr__(self, "frame_generator", frame_generator)
        scan_generator = timspeak.data_handlers.sample_iterator.ImNeighborGenerator(
            dia_data = self.dia_data,
            im_tolerance=self.im_tolerance,
        )
        object.__setattr__(self, "scan_generator", scan_generator)
        ion_pair_generator = timspeak.data_handlers.sample_iterator.MZIonPairGenerator(
            dia_data = self.dia_data,
            ppm_tolerance=self.ppm_tolerance,
        )
        object.__setattr__(self, "ion_pair_generator", ion_pair_generator)
        object.__setattr__(self, "scans_per_frame", self.dia_data.cycle.shape[-2])


    def smooth_all_scans(self) -> np.ndarray:
        """
        Smooths all the scans in the DIA data.

        Returns:
        - np.ndarray: An array containing the smoothed intensity values.
        """
        buffer_array = np.zeros_like(self.dia_data.intensity_values)
        timspeak.performance_utilities.multiprocessing.parallel(self.smooth_scan)(
            range(len(self.dia_data.tof_indptr) - 1),
            buffer_array,
        )
        return buffer_array

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def smooth_scan(
            self,
            scan_index: int,
            smooth_intensity_values: np.ndarray
    ) -> None:
        """
        Smooths a single scan.

        Parameters:
        - scan_index (int): The index of the scan to be smoothed.
        - smooth_intensity_values (np.ndarray): An array to store the smoothed intensity values.

        Returns:
        - None
        """
        if self.is_empty_scan(scan_index):
            return
        for other_rt_scan in self.frame_generator.from_scan(scan_index):
            self.get_smooth_values_from_scan_generator(
                 scan_index,
                 other_rt_scan,
                 smooth_intensity_values
            )

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_smooth_values_from_scan_generator(
         self,
         scan_index: int,
         other_rt_scan,
         smooth_intensity_values: np.ndarray,
    ) -> None:
        """
        Retrieves the smoothed intensity values from the scan generator.

        Parameters:
        - scan_index (int): The index of the current scan.
        - other_rt_scan: The neighboring scan obtained from the scan generator.
        - smooth_intensity_values (np.ndarray): An array to store the smoothed intensity values.

        Returns:
        - None
        """
        for other_im_scan in self.scan_generator.from_scan(other_rt_scan):
            if self.is_empty_scan(other_im_scan):
                continue
            scan_correction = self.calculate_scan_correction(
                scan_index,
                other_im_scan
            )
            self.get_smooth_values_from_ion_pair_generator(
                scan_index,
                other_im_scan,
                scan_correction,
                smooth_intensity_values
            )

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def get_smooth_values_from_ion_pair_generator(
         self,
         scan_index: int,
         other_im_scan,
         scan_correction,
         smooth_intensity_values: np.ndarray,
    ) -> None:
        """
        Retrieves the smoothed intensity values from the ion pair generator.

        Parameters:
        - scan_index (int): The index of the current scan.
        - other_im_scan: The neighboring scan obtained from the ion pair generator.
        - scan_correction: The scan correction factor.
        - smooth_intensity_values (np.ndarray): An array to store the smoothed intensity values.

        Returns:
        - None
        """
        for index1, index2 in self.ion_pair_generator.from_scan_pair(
             scan_index,
             other_im_scan
        ):
            tof_correction = self.calculate_tof_correction(
                index1,
                index2
            )
            corrected_intensity = scan_correction * tof_correction
            corrected_intensity *= self.dia_data.intensity_values[index2]
            smooth_intensity_values[index1] += corrected_intensity

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def is_empty_scan(self, scan_index: int) -> bool:
        """
        Checks if a scan is empty.

        Parameters:
        - scan_index (int): The index of the scan to check.

        Returns:
        - bool: True if the scan is empty, False otherwise.
        """
        start = self.dia_data.tof_indptr[scan_index]
        end = self.dia_data.tof_indptr[scan_index + 1]
        return start == end

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def calculate_scan_correction(self, scan1: int, scan2: int):
        """
        Calculates the scan correction factor.

        Parameters:
        - scan1 (int): The index of the first scan.
        - scan2 (int): The index of the second scan.

        Returns:
        - The scan correction factor.
        """
        scan_index_im1 = scan1 % self.scans_per_frame
        frame_index_rt1 = scan1 // self.scans_per_frame
        scan_index_im2 = scan2 % self.scans_per_frame
        frame_index_rt2 = scan2 // self.scans_per_frame
        im1 = self.dia_data.im_values[scan_index_im1]
        rt1 = self.dia_data.rt_values[frame_index_rt1]
        im2 = self.dia_data.im_values[scan_index_im2]
        rt2 = self.dia_data.rt_values[frame_index_rt2]
        im_correction = self.gauss_correction(im1 - im2, self.im_sigma)
        rt_correction = self.gauss_correction(rt1 - rt2, self.rt_sigma)
        return im_correction * rt_correction

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def calculate_tof_correction(self, scan1: int, scan2: int):
        return 1

    @timspeak.performance_utilities.compiling.njit(nogil=True)
    def gauss_correction(self, x=0, sigma=1):
        """
        Calculates the Gaussian correction factor.

        Parameters:
        - x (float): The input value.
        - sigma (float): The standard deviation of the Gaussian.

        Returns:
        - The Gaussian correction factor.
        """
        if sigma == 0:
            return 1
        else:
            return np.exp(-(x / sigma)**2 / 2)