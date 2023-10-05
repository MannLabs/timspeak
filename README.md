# TimsPeak

TimsPeak is an open-source Python package to detect and centroid peaks in TIMS-TOF data.

## TOC

- [TimsPeak](#timspeak)
  - [TOC](#toc)
  - [License](#license)
  - [Installation](#installation)
  - [How it works](#how-it-works)
    - [Smoothing](#smoothing)
    - [Clustering](#clustering)
    - [Statistical metrics](#statistical-metrics)
    - [Deisotoping](#deisotoping)
    - [Parallelization](#parallelization)
    - [RAM](#ram)
  - [Usage](#usage)
    - [CLI](#cli)
    - [Configuration file](#configuration-file)

## License

TODO.

## Installation

Timspeak can be installed in editable mode. This allows to fully customize the software and modify the source code to different specific needs. Once a location has been chosen where to install Timspeak, navigate into the folder and copy the Timspeak package into it.

It is highly recommended to use a conda virtual environment in order to avoid having dependency problems with different packages that might be installed in the user's system. A new conda environment can manually be created by e.g.:

```{bash}
conda create -n timspeak python=3.10 -y
```

Once a new conda has been created, it can be activated by:

```{bash}
conda activate timspeak
```

Now Timspeak and all its dependancies need to be installed. To take advantage of all features and allow development (with the -e flag) use:

```{bash}
pip install -e "./timspeak[development]"
```

By using the editable flag -e, all modifications to the Timspeak source code folder are directly reflected when running Timspeak.

## How it works

The peak picking process involves identifying and extracting the peaks from the acquired mass spectrometry data. The fundamental peak picker process involves three main stages: data smoothing, peak clustering and deisotoping. Timspeak implements custom functions for these stages and goes further by including additional functionalities to generate statistical metrics that help to improve the deisotoping process. Each stage is described in more detail in the following subsections.

### Smoothing
Smoothing is the initial stage of the peak picking process. The acquired mass spectrometry data often contains noise, baseline variations, and other unwanted fluctuations. A custom smoothing algorithm is applied to reduce noise and enhance peak detection. Timspeak applies to each scan in IM (Ion Mobility) and RT (Retention Time) axis a Gaussian smoothing that helps to improve peak detection accuracy by reducing false positive peaks caused by noise. On the mz axis, this smoothing is uniform instead of Gaussian.

### Clustering
After smoothing, the data is processed to identify clusters of data points that correspond to potential peaks. Clustering algorithms group nearby data points based on their intensity values and proximity in the IM, RT and mass-to-charge ratio (m/z) domain. Timspeak implements a custom clustering algorithm that takes the highest intensity point (apex) detected and from scan down and group together all the remaining lower intensity points (within a region defined by a given IM and RT thresholds) in the same cluster.

### Statistical metrics
Timspeak provides useful functionalities to extract statistical metrics from clusters. Among these metrics can be found: a cluster size calculator to get the number of peaks in a cluster, MZ, IM and RT calculators to get the weighted mean of m/z, IM and RT respectively, and IM boundary and RT boundary calculators to get the minimum and maximum values of IM and RT respectively to define the clusters' boundaries. Finally, the IM and RT projections (i.e. ion mobilogram and XIC) are store explicitly for each peak.

### Deisotoping
Deisotoping is the process of identifying and merging isotopic peaks within a cluster. Isotopic peaks arise due to the presence of different isotopes of an element in the sample. Deisotoping algorithms analyze the relative intensities and spacing patterns of peaks within a cluster to identify the monoisotopic peak (the peak corresponding to the most abundant isotope). By removing isotopic peaks, the deisotoping stage helps to simplify the peak list and reduce redundancy, improving downstream analysis. Timspeak processes every single scan by comparing it with other scans in the same RT and IM isolation window. Then, Timspeak identifies all peaks, from the previous isolation window, that are within the ppm tolerance and stores a pointer to the peak with largest MZ, clearing every other pek. After deisotoping, Timspeak applies the Kolomgorov-Smirnov test to 1D projections scans on the RT and IM axis for charge 2 and charge 3 isotopes. Timspeak also applies to Kolmogorov-Smirnov test to isotopic peak cluster adding an extra filtering layer for cluster refinement.

These stages are crucial in the peak picking process as they enable the accurate extraction of peaks from mass spectrometry data. By applying smoothing, clustering, and deisotoping algorithms, and the Kolmogorov-Smirnov test as an extra filetering cluster stage Timspeak can effectively identify and characterize the peaks, providing valuable information for subsequent data analysis.

### Parallelization

Timspeak uses a custom module that provides utility functions for parallel processing using threads and multiprocessing in Python that speed up a sample processing time.

The multiprocessing module has the following dependencies: tqdm, numba and numpy.

In order to exploit parallelism the user only needs to specify the number of threads to use in the configuration file (larger than 1). It is recommended to use the larger prime number of threads available in the user's system.

### RAM

Timspeak uses a custom module for creation of temporary memory-mapped (mmapped) arrays in Python. Once an array is stored to disk, a memory map is created to access directly that disk memory region reading the stored data. This way Timspeak dramatically reduces its RAM consumption freeing RAM for other uses and enabling itself to handle datasets' sizes that could crash the user's system otherwise.

## Usage

### CLI

Usage for "timspeak" Command Line Tool.

Previous to actually using the Timspeak package, activate the conda environment used for timspeak installation:

```{bash}
$conda activate timspeak
```

Once the conda environment is activated the "timspeak" basic command line tool functionalities can be displayed by:

```{bash}
$peackpicker

********************
* timspeak 0.0.1 *
********************
Usage: timspeak [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  run_pipeline  Run timspeak execution_pipeline.
```

In order to show the timspeak version number, the following must be typed:

```{bash}
$timspeak -v
timspeak, version 0.0.1
```

The command run_pipeline is used for starting the execution pipeline of peak picker, options and arguments can be
displayed by typing:

```{bash}
$timspeak run_pipeline
********************
* timspeak 0.0.1 *
********************
Usage: timspeak run_pipeline [OPTIONS] CONFIGFILE [SAMPLEFILE] [OUTPUTFILE]

  Run timspeak execution_pipeline.

Options:
  -h, --help  Show this message and exit.
```

In order to run the pipeline a configuration file in JSON or YAML format needs to be specified as follows:

```{bash}
$ timspeak run_pipeline timspeak/configuration_files/default_configuration.json
********************
* timspeak 0.0.1 *
********************
WARNING:root:WARNING: Temp mmap arrays are written to /tmp/temp_mmap_2s82kcym. Cleanup of this folder is
OS dependant, and might need to be triggered manually! Current space: 330,940,342,272
<class 'logging.RootLogger'>
2023-06-08 09:54:02> ---------- INITIALIZE LOGGER ----------
2023-06-08 09:54:02> ---------- PLATFORM INFO----------
2023-06-08 09:54:02> platform:        Linux-5.15.0-73-generic-x86_64-with-glibc2.31
2023-06-08 09:54:02> system:          Linux
2023-06-08 09:54:02> release:         5.15.0-73-generic
```

As the examples shows the Timspeak package contains a JSON and YAML examples of these files with default values in
the timspeak/configuration_files/ directory. A more in detail information of the configuration file is provided
in next subsection.

Additionaly, Timspeak provides the functionality to override the sample and output file names specified in the
configuration file by using the arguments [SAMPLEFILE] and [OUTPUTFILE]. In order to override them a sample and an
output file names must be specified at the command line:

```{bash}
$ timspeak run_pipeline configuration.json my_config.yaml my_output.hdf
```

If only one the arguments is specified Timspeak will take it as the sample file name. If the output file name needs to be set from command line, then the two arguments must be provided.

### Configuration file

Timspeak requires a configuration file in order to run its entire pipeline. The configuration file specifies all the values for the required variables needed to read the sample data set, set the parameters for the different algorithms included in Timspeak and write the output. The configuration file can be written in two formats: JSON and YAML. The directory "timspeak/configuration_files" contain an example of a configuration file with optimal default values in both formats. Below is a detailed description of each section and its corresponding parameters:

General Information

sample_file_name: The name of the sample file being processed (it should be a .d folder).
output_file_name: The name of the output file generated by Timspeak. Timspeak accepts two output formats, H5DF and
Zarr (https://zarr.readthedocs.io/en/stable/index.html).
number_of_threads: The number of threads or parallel processes to be used for the task.

* **Smoothing**
  * **algorithm_name**: The name of the smoothing algorithm used for data smoothing.
  * **im_sigma**: standard deviation for gaussian correction in the Ion Mobility (IM) axis.
  * **im_tolerance**: value used to set the boundaries of the isolation window in IM axis.
  * **ppm_tolerance**: value used to set the upper and lower limits of the tof indices.
  * **rt_sigma**: standard deviation for gaussian correction in the retention time (RT) axis.
  * **rt_tolerance**: value used to set the boundaries of the isolation window in RT axis.
* **Clustering**
  * **algorithm_name**: The name of the clustering algorithm used for data clustering.
  * **im_tolerance**: value used to set the boundaries of the isolation window in IM axis.
  * **ppm_tolerance**: value used to set the upper and lower limits of the tof indices.
  * **rt_tolerance**: value used to set the boundaries of the isolation window in RT axis.
  * **clustering_threshold**: This is the minimum number of points required to consider a set of points to be a cluster.
* **MS1 - precursors**
  * **min_size**: The minimum size or intensity threshold for precursor data. Precursors with intensity below this threshold will be filtered out. The specified value is 10.
  * **MS1 - isotopes - charge 2**
    * **im_tolerance**: value used to set the boundaries of the isolation window in IM axis of charge 2 precursors.
    * **ppm_tolerance**: value used to set the upper and lower limits of the tof indices of 2 precursors.
    * **rt_tolerance**: value used to set the boundaries of the isolation window in RT axis of 2 precursors.
  * **MS1 - isotopes - charge 3**
    * **im_tolerance**: value used to set the boundaries of the isolation window in IM axis of charge 3 precursors.
    * **ppm_tolerance**: value used to set the upper and lower limits of the tof indices of 3 precursors.
    * **rt_tolerance**: value used to set the boundaries of the isolation window in RT axis of 3 precursors.
  * **MS1 - isotopes - monoisotopic_precursors**
    * **ks_2d_threshold**: this values is used to filter all monoisotopic precursors.
* **MS2 - fragments**
  * **min_size**: The minimum number of fragments per precursor.
