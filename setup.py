from setuptools import setup, find_packages
import os
import sys
from shutil import copyfile

install_requires = ["numpy", "scipy", "numba", "tables", "matplotlib", "pyyaml"]

if not os.path.exists("tmps"):
    os.makedirs("tmps")
copyfile("puex/version.py", "tmps/version.py")
__import__("tmps.version")
package = sys.modules["tmps"]
package.version.update_release_version("puex")


setup(
    name="puex",
    version=package.version.get_version(pep440=True),
    description="Using the non-negative least squares method "
    "to extract series of pulses from waveforms",
    author="Samuel Flis",
    author_email="samuel.flis@desy.de",
    url="https://github.com/sflis/SSM-analysis",
    packages=find_packages(),
    provides=["puex"],
    license="GNU Lesser General Public License v3 or later",
    install_requires=install_requires,
    extras_requires={},
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            # 'ssdaq = ssdaq.bin.ssdaqd:main',
        ]
    },
)
