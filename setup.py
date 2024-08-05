#!/usr/bin/env python

from setuptools import setup

NAME = "trispectral"
MAINTAINER = ""
MAINTAINER_EMAIL = "velizhaninae@gmail.com"
DESCRIPTION = ""
LICENSE = ""
VERSION = "1.0.0"
LONG_DESCRIPTION = ""

NUMPY_MIN_VERSION = "1.26.4"
SCIPY_MIN_VERSION = "1.11.3"

min_dependent_packages = {
    "numpy": NUMPY_MIN_VERSION,
    "scipy": SCIPY_MIN_VERSION,
}


def setup_package():
    python_requires = ">=3.12"

    metadata = dict(
        name=NAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        #        download_url=DOWNLOAD_URL,
        #        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
        python_requires=python_requires,
        install_requires=[
            f"{p} >= {v}" for (p, v) in min_dependent_packages.items()
        ],
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
