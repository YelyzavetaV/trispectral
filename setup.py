#!/usr/bin/env python

from setuptools import setup

NAME = "trispectral"
MAINTAINER = ""
MAINTAINER_EMAIL = "velizhaninae@gmail.com"
DESCRIPTION = ""
LICENSE = ""
VERSION = "1.0.0"
LONG_DESCRIPTION = ""


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
            "numpy >= 1.26.4",
            "scipy >= 1.11.3",
            "pytest",
            "matplotlib",
            "pandas",
        ],
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
