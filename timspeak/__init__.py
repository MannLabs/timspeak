#!python


__package__ = "timspeak."
__project__ = "Peak Picker"
__version__ = "0.0.1"
__license__ = "Apache 2.0"
__description__ = "Package for peak detecting in TIMS-TOF data"
__author__ = "Sander Willems"
__author_email__ = "todo@todo.todo"
__github__ = "https://todo.todo"
__keywords__ = [
    "bioinformatics",
]
__python_version__ = ">=3.8,<4"
__classifiers__ = [
    # "Development Status :: 1 - Planning",
    # "Development Status :: 2 - Pre-Alpha",
    "Development Status :: 3 - Alpha",
    # "Development Status :: 4 - Beta",
    # "Development Status :: 5 - Production/Stable",
    # "Development Status :: 6 - Mature",
    # "Development Status :: 7 - Inactive"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "timspeak=timspeak.cli:run",
]
__urls__ = {
    "GitHub": __github__,
    # "ReadTheDocs": None,
    # "PyPi": None,
    # "Scientific paper": None,
}
